use crate::{
    CallHook, Error, Instance, Store,
    engine::{
        CodeView, EngineFunc, LiftFromCells, LowerToCells,
        executor::handler::{
            dispatch::{ExecutionOutcome, execute_until_done},
            state::{Freg32, Freg64, Inst, Ip, Ireg, Sp, Stack, VmState},
            utils::{self, resolve_instance},
        },
    },
    func::HostFuncEntity,
    ir::{BoundedSlotSpan, Slot, SlotSpan},
    store::{CallHooks, StoreError},
};
use core::marker::PhantomData;

pub struct WasmFuncCall<'a, T, State> {
    store: &'a mut Store<T>,
    stack: &'a mut Stack,
    code: CodeView<'a>,
    callee_ip: Ip,
    callee_sp: Sp,
    instance: Inst,
    state: State,
    ireg: Ireg,
    freg32: Freg32,
    freg64: Freg64,
    /// majit JIT tier handle if this function is eligible: `(cache_key,
    /// num_slots, writes_result, action)`, where `cache_key = ops.as_ptr()`
    /// indexes the per-function cache (prepassed once at call init), `writes_result`
    /// is false for a no-result function, and `action` is this call's adaptive tier
    /// decision. `None` -> stock executor (ineligible or majit disabled).
    #[cfg(feature = "majit-jit")]
    majit: Option<(usize, usize, bool, super::majit::kernel::TierAction)>,
}

impl<'a, T, State> WasmFuncCall<'a, T, State> {
    fn new_state<NewState>(self, state: NewState) -> WasmFuncCall<'a, T, NewState> {
        WasmFuncCall {
            store: self.store,
            stack: self.stack,
            code: self.code,
            callee_ip: self.callee_ip,
            callee_sp: self.callee_sp,
            instance: self.instance,
            state,
            ireg: self.ireg,
            freg32: self.freg32,
            freg64: self.freg64,
            #[cfg(feature = "majit-jit")]
            majit: self.majit,
        }
    }
}

mod state {
    use super::Sp;
    use crate::{engine::InOutParams, func::Trampoline};
    use core::marker::PhantomData;

    pub type Uninit = PhantomData<marker::Uninit>;
    pub type Init = PhantomData<marker::Init>;
    pub type Resumed = PhantomData<marker::Resumed>;

    mod marker {
        pub enum Uninit {}
        pub enum Init {}
        pub enum Resumed {}
    }

    pub struct UninitHost<'a> {
        pub sp: Sp,
        pub inout: InOutParams<'a>,
        pub trampoline: Trampoline,
    }

    pub struct InitHost<'a> {
        pub sp: Sp,
        pub inout: InOutParams<'a>,
        pub trampoline: Trampoline,
    }

    pub trait Execute {}
    impl Execute for Init {}
    impl Execute for Resumed {}
    pub struct Done {
        pub sp: Sp,
    }
}

impl<'a, T> WasmFuncCall<'a, T, state::Uninit> {
    pub fn write_params<Params>(self, params: Params) -> WasmFuncCall<'a, T, state::Init>
    where
        Params: LowerToCells,
    {
        let mut sp = self.callee_sp;
        let Ok(_) = params.lower_to_cells(&*self.store, &mut sp) else {
            panic!("failed to write parameter values to cells")
        };
        self.new_state(PhantomData)
    }
}

impl<'a, T, State: state::Execute> WasmFuncCall<'a, T, State> {
    pub fn execute(mut self) -> Result<WasmFuncCall<'a, T, state::Done>, ExecutionOutcome> {
        self.store.invoke_call_hook(CallHook::CallingWasm)?;
        let outcome = self.execute_until_done();
        self.store.invoke_call_hook(CallHook::ReturningFromWasm)?;
        let sp = outcome?;
        Ok(self.new_state(state::Done { sp }))
    }

    fn execute_until_done(&mut self) -> Result<Sp, ExecutionOutcome> {
        // `self.majit` is `Some` only when majit is enabled and the function is
        // eligible (decided once at call init, carrying this call's tier action).
        #[cfg(feature = "majit-jit")]
        if let Some((key, num_slots, writes_result, action)) = self.majit {
            return self.execute_majit(key, num_slots, writes_result, action);
        }
        self.execute_stock()
    }

    /// Run this call on the stock handler-threaded executor.
    fn execute_stock(&mut self) -> Result<Sp, ExecutionOutcome> {
        let store = self.store.prune();
        let (mem0, mem0_len) = utils::extract_mem0(store, self.instance);
        let mut state = VmState::new(store, self.stack, self.code);
        execute_until_done(
            &mut state,
            self.callee_ip,
            self.callee_sp,
            mem0,
            mem0_len,
            self.instance,
            self.ireg,
            self.freg32,
            self.freg64,
        )
    }

    /// Run an eligible function, choosing the JIT tier or the stock executor by
    /// the function's adaptive [`tier`](super::majit::kernel::tier_action)
    /// policy. While probing, the chosen path is timed so the policy can commit
    /// to whichever tier is faster for this function's actual per-call work.
    #[cfg(feature = "majit-jit")]
    fn execute_majit(
        &mut self,
        key: usize,
        num_slots: usize,
        writes_result: bool,
        action: super::majit::kernel::TierAction,
    ) -> Result<Sp, ExecutionOutcome> {
        use super::majit::kernel::{self, TierAction};
        match action {
            TierAction::Stock => self.execute_stock(),
            TierAction::Jit => self.run_jit(key, num_slots, writes_result),
            TierAction::ProbeJit => {
                let t = std::time::Instant::now();
                let r = self.run_jit(key, num_slots, writes_result);
                kernel::record_probe_jit(key, t.elapsed().as_nanos() as u64);
                r
            }
            TierAction::ProbeStock => {
                let t = std::time::Instant::now();
                let r = self.execute_stock();
                kernel::record_probe_stock(key, t.elapsed().as_nanos() as u64);
                r
            }
        }
    }

    /// Run an eligible function on the majit JIT tier.
    ///
    /// Seeds the kernel's cell array from the frame slots, runs the
    /// majit-traced mainloop on the thread's persistent driver (which reuses the
    /// function's compiled loop across calls), writes the single result back to
    /// slot 0 (the wasm return convention) unless the function is no-result, and
    /// returns the frame base — matching the stock executor's
    /// `DoneReason::Return(callee_sp)`.
    #[cfg(feature = "majit-jit")]
    fn run_jit(
        &mut self,
        key: usize,
        num_slots: usize,
        writes_result: bool,
    ) -> Result<Sp, ExecutionOutcome> {
        // Allocate the real frame slots plus the prepass's reserved scratch slots
        // (used for operand pre-materialization); only the real slots are seeded
        // from the caller's frame, the scratch tail stays zero.
        let mut slots = alloc::vec![0i64; num_slots + super::majit::prepass::NUM_SCRATCH];
        for (i, slot) in slots.iter_mut().take(num_slots).enumerate() {
            *slot = unsafe { self.callee_sp.get::<i64>(Slot::from(i as u16)) };
        }
        // The per-run table of raw pointers to the instance's globals (only built
        // when the function references globals). Kept alive across the run; the
        // kernel's residual global helpers read it through `GLOBALS_CTX`.
        let globals_table = if super::majit::kernel::key_uses_globals(key) {
            utils::resolve_globals_table(self.store.prune(), self.instance)
        } else {
            alloc::vec::Vec::new()
        };
        // The default linear memory base/len, threaded to the kernel so a memory
        // load resolves through its residual helper. Re-read each call so a
        // `memory.grow` relocation is reflected rather than baked into the trace.
        let (mem0, mem0_len) = utils::extract_mem0(self.store.prune(), self.instance);
        let result = super::majit::kernel::run_persistent(
            key,
            &slots,
            mem0.addr() as i64,
            mem0_len.get() as i64,
            globals_table.as_ptr(),
            globals_table.len(),
        );
        // A residual access that trapped (out-of-bounds memory, or a trapping
        // f64→int conversion) cannot trap from inside the kernel; it flags the trap
        // and we surface it here.
        if super::majit::kernel::take_mem_trap() {
            if super::majit::kernel::take_mem_did_store() {
                // The JIT run already committed stores (in program order, up to the
                // first trapping access). Re-running stock would double-apply them,
                // so raise the trap directly — memory already matches what stock
                // would leave at the trap point. The recorded code is the first
                // trap in program order (out-of-bounds, or a trapping conversion).
                return Err(ExecutionOutcome::from(
                    super::majit::kernel::take_trap_code(),
                ));
            }
            // No store was committed (pure-load run, or a store that trapped before
            // its first write): the discarded run left memory untouched, so re-run
            // on the stock executor for the faithful wasm trap.
            return self.execute_stock();
        }
        // A no-result function reserves no result slot; writing slot 0 would clobber
        // memory past a zero-slot frame, so only write a result when the function
        // returns one.
        if writes_result {
            unsafe { self.callee_sp.set::<i64>(Slot::from(0), result) };
        }
        Ok(self.callee_sp)
    }
}

impl<'a, T> WasmFuncCall<'a, T, state::Resumed> {
    pub fn provide_host_results<Params>(
        self,
        params: Params,
        slots: SlotSpan,
    ) -> WasmFuncCall<'a, T, state::Init>
    where
        Params: LowerToCells,
    {
        let mut sp = self.callee_sp.offset(slots.head());
        let Ok(_) = params.lower_to_cells(&*self.store, &mut sp) else {
            panic!("failed to store provided host results to cells")
        };
        self.new_state(PhantomData)
    }
}

impl<'a, T> WasmFuncCall<'a, T, state::Done> {
    pub fn write_results<Results>(self, results: Results) -> Results::Value
    where
        Results: LiftFromCells,
    {
        let mut sp = self.state.sp;
        let Ok(value) = results.lift_from_cells(&*self.store, &mut sp) else {
            panic!("failed to load result values from cells")
        };
        value
    }
}

pub fn init_wasm_func_call<'a, T>(
    store: &'a mut Store<T>,
    code: CodeView<'a>,
    stack: &'a mut Stack,
    func: EngineFunc,
    instance: Instance,
) -> Result<WasmFuncCall<'a, T, state::Uninit>, Error> {
    let Some(compiled_func) = code.get_or_compile(Some(store.inner.fuel_mut()), func)? else {
        panic!("missing function entry at: {func:?}")
    };
    let ops = compiled_func.ops();
    let callee_ip = Ip::from(ops);
    let len_local_slots = compiled_func.len_local_slots();
    let len_stack_slots = compiled_func.len_stack_slots();
    // Prepass the function to a majit MiniProgram once (cached by op-stream
    // pointer); `None` (any op outside the supported subset) falls back to the
    // stock executor. `majit` carries `(cache_key, num_slots)` when eligible.
    #[cfg(feature = "majit-jit")]
    let majit = super::majit::kernel::ensure_cached(ops, len_local_slots, len_stack_slots).map(
        |(num_slots, writes_result, action)| {
            (ops.as_ptr() as usize, num_slots, writes_result, action)
        },
    );
    // Note: using a length of 0 for `callee_params` simply has the effect that all frame
    //       cells are initialized to zero which is a safe default. There currently is not
    //       an easy and efficient way to get the number of parameter cells at this point
    //       so we simply default to 0.
    let callee_params = BoundedSlotSpan::new(SlotSpan::new(Slot::from(0)), 0);
    let instance = resolve_instance(store.prune(), &instance).into();
    let callee_sp = stack.push_frame(
        None,
        callee_ip,
        callee_params,
        len_local_slots,
        len_stack_slots,
        Some(instance),
    )?;
    let (ireg, freg32, freg64) = stack.regs();
    Ok(WasmFuncCall {
        store,
        stack,
        code,
        callee_ip,
        callee_sp,
        instance,
        state: PhantomData,
        ireg,
        freg32,
        freg64,
        #[cfg(feature = "majit-jit")]
        majit,
    })
}

pub fn resume_wasm_func_call<'a, T>(
    store: &'a mut Store<T>,
    code: CodeView<'a>,
    stack: &'a mut Stack,
) -> Result<WasmFuncCall<'a, T, state::Resumed>, Error> {
    let (callee_ip, callee_sp, instance, ireg, freg32, freg64) = stack.restore_frame();
    Ok(WasmFuncCall {
        store,
        stack,
        code,
        callee_ip,
        callee_sp,
        instance,
        state: PhantomData,
        ireg,
        freg32,
        freg64,
        // The resume path continues an already-running call; the majit tier only
        // attempts fresh `init_wasm_func_call`s.
        #[cfg(feature = "majit-jit")]
        majit: None,
    })
}

pub fn init_host_func_call<'a, T>(
    store: &'a mut Store<T>,
    stack: &'a mut Stack,
    func: HostFuncEntity,
) -> Result<HostFuncCall<'a, T, state::UninitHost<'a>>, Error> {
    let len_param_cells = func.len_param_cells();
    let len_result_cells = func.len_result_cells();
    let trampoline = *func.trampoline();
    let callee_params = BoundedSlotSpan::new(SlotSpan::new(Slot::from(0)), len_param_cells);
    let (sp, inout) = stack.prepare_host_frame(None, callee_params, len_result_cells)?;
    Ok(HostFuncCall {
        store,
        state: state::UninitHost {
            sp,
            inout,
            trampoline,
        },
    })
}

#[derive(Debug)]
pub struct HostFuncCall<'a, T, State> {
    store: &'a mut Store<T>,
    state: State,
}

impl<'a, T> HostFuncCall<'a, T, state::UninitHost<'a>> {
    pub fn write_params<Params>(self, params: Params) -> HostFuncCall<'a, T, state::InitHost<'a>>
    where
        Params: LowerToCells,
    {
        let state::UninitHost {
            sp,
            inout,
            trampoline,
        } = self.state;
        let mut sp_writer = sp;
        let Ok(_) = params.lower_to_cells(&*self.store, &mut sp_writer) else {
            panic!("failed to store parameter values to cells")
        };
        HostFuncCall {
            store: self.store,
            state: state::InitHost {
                sp,
                inout,
                trampoline,
            },
        }
    }
}

impl<'a, T> HostFuncCall<'a, T, state::InitHost<'a>> {
    pub fn execute(self) -> Result<HostFuncCall<'a, T, state::Done>, Error> {
        let state::InitHost {
            sp,
            inout,
            trampoline,
        } = self.state;
        let outcome = self
            .store
            .prune()
            .call_host_func(trampoline, None, inout, CallHooks::Ignore);
        if let Err(error) = outcome {
            match error {
                StoreError::External(error) => return Err(error),
                StoreError::Internal(error) => panic!("internal interpreter error: {error}"),
            }
        }
        Ok(HostFuncCall {
            store: self.store,
            state: state::Done { sp },
        })
    }
}

impl<'a, T> HostFuncCall<'a, T, state::Done> {
    pub fn write_results<Results>(self, results: Results) -> Results::Value
    where
        Results: LiftFromCells,
    {
        let mut sp = self.state.sp;
        let Ok(value) = results.lift_from_cells(&*self.store, &mut sp) else {
            panic!("failed to load result value from cells")
        };
        value
    }
}
