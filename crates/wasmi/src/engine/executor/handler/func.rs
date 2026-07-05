use crate::{
    CallHook, Error, Func, Instance, Store,
    engine::{
        CodeView, EngineFunc, LiftFromCells, LowerToCells,
        executor::handler::{
            dispatch::{ExecutionOutcome, execute_until_done},
            state::{Freg32, Freg64, Inst, Ip, Ireg, Sp, Stack, VmState},
            utils::{self, resolve_instance},
        },
    },
    func::{FuncEntity, HostFuncEntity},
    ir::{BoundedSlotSpan, Slot, SlotSpan},
    store::{CallHooks, StoreError},
};
use core::marker::PhantomData;

/// TLS cache for the callee execution Stack, reused across `run_jit` calls
/// to avoid repeated allocation.
#[cfg(feature = "majit-jit")]
std::thread_local! {
    static CALLEE_STACK_CACHE: core::cell::RefCell<Option<Stack>> =
        core::cell::RefCell::new(None);
}

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
    /// num_slots, writes_result, action, slot_map)`, where `cache_key = ops.as_ptr()`
    /// indexes the per-function cache (prepassed once at call init), `writes_result`
    /// is false for a no-result function, `action` is this call's adaptive tier
    /// decision, and `slot_map` maps dense slot indices to original frame slots.
    /// `None` -> stock executor (ineligible or majit disabled).
    #[cfg(feature = "majit-jit")]
    majit: Option<(
        usize,
        usize,
        bool,
        super::majit::kernel::TierAction,
        alloc::vec::Vec<u16>,
    )>,
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
        if let Some((key, num_slots, writes_result, action, ref slot_map)) = self.majit {
            let slot_map = slot_map.clone();
            return self.execute_majit(key, num_slots, writes_result, action, &slot_map);
        }
        self.execute_stock()
    }

    /// Run this call on the stock executor starting at a specific instruction
    /// pointer (not the function's entry point). Used by the yield-to-stock
    /// pathway when the kernel has already executed the prefix of the function
    /// and committed its side effects.
    #[cfg(feature = "majit-jit")]
    fn execute_stock_at(&mut self, ip: Ip) -> Result<Sp, ExecutionOutcome> {
        let store = self.store.prune();
        let (mem0, mem0_len) = utils::extract_mem0(store, self.instance);
        let mut state = VmState::new(store, self.stack, self.code);
        execute_until_done(
            &mut state,
            ip,
            self.callee_sp,
            mem0,
            mem0_len,
            self.instance,
            self.ireg,
            self.freg32,
            self.freg64,
        )
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
        slot_map: &[u16],
    ) -> Result<Sp, ExecutionOutcome> {
        use super::majit::kernel::{self, TierAction};
        match action {
            TierAction::Stock => self.execute_stock(),
            TierAction::Jit => self.run_jit(key, num_slots, writes_result, slot_map),
            TierAction::ProbeJit => {
                let t = std::time::Instant::now();
                let r = self.run_jit(key, num_slots, writes_result, slot_map);
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
        slot_map: &[u16],
    ) -> Result<Sp, ExecutionOutcome> {
        // Allocate the dense slots plus the prepass's reserved scratch slots
        // (used for operand pre-materialization); seed each dense slot from the
        // original frame position via slot_map, the scratch tail stays zero.
        let mut slots = alloc::vec![0i64; num_slots + super::majit::prepass::NUM_SCRATCH];
        for (dense_idx, &orig) in slot_map.iter().enumerate() {
            slots[dense_idx] = unsafe { self.callee_sp.get::<i64>(Slot::from(orig)) };
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
        // Register the call runner so the kernel's call_internal_residual
        // can execute CallInternal instructions via the stock executor.
        // Take the cached callee Stack from TLS (or create one). Avoids
        // re-allocating a Stack on every run_jit call.
        let callee_stack = CALLEE_STACK_CACHE.with(|c| {
            c.borrow_mut().take().unwrap_or_else(|| {
                Stack::new(&crate::engine::limits::StackConfig::default())
            })
        });
        let mut call_ctx = CallRunnerCtx {
            store: self.store.prune() as *mut crate::store::PrunedStore,
            code: self.code,
            instance: self.instance,
            callee_stack,
        };
        super::majit::kernel::set_call_runner(
            call_runner_fn,
            &mut call_ctx as *mut CallRunnerCtx as *mut (),
        );
        let result = super::majit::kernel::run_persistent(
            key,
            &slots,
            mem0.addr() as i64,
            mem0_len.get() as i64,
            globals_table.as_ptr(),
            globals_table.len(),
        );
        super::majit::kernel::clear_call_runner();
        // Return the callee Stack to the TLS cache for reuse.
        CALLEE_STACK_CACHE.with(|c| { *c.borrow_mut() = Some(call_ctx.callee_stack); });
        // The kernel yielded at a CallInternal. Flush the kernel's computed
        // slots to the real frame and resume the stock executor AT that
        // instruction — not from byte 0. This avoids double-applying side
        // effects the kernel already committed via residual calls.
        if super::majit::kernel::take_yield_to_stock() {
            // A yield after a residual trap is not safe: the kernel continued
            // with dummy values after the trap and the flushed slots may be
            // corrupted. Fall back to the trap/stock-from-start path instead.
            if super::majit::kernel::take_mem_trap() {
                if super::majit::kernel::take_mem_did_store() {
                    return Err(ExecutionOutcome::from(
                        super::majit::kernel::take_trap_code(),
                    ));
                }
                return self.execute_stock();
            }
            let flushed = super::majit::kernel::take_yield_slots();
            let byte_offset = super::majit::kernel::take_yield_offset() as usize;
            // Flush kernel slots to the real frame so the CallInternal handler
            // reads correct parameter values from the frame.
            for (i, &val) in flushed.iter().enumerate() {
                unsafe { self.callee_sp.set::<i64>(Slot::from(i as u16), val) };
            }
            // Resume the stock executor at the CallInternal instruction.
            let yield_ip = unsafe { self.callee_ip.add(byte_offset) };
            return self.execute_stock_at(yield_ip);
        }
        // The kernel hit a tail call it cannot handle. Fall back to the stock
        // executor if no stores were committed; otherwise the stock re-run
        // would double-apply stores.
        if super::majit::kernel::take_bail_to_stock() {
            if !super::majit::kernel::take_mem_did_store() {
                return self.execute_stock();
            }
            // Stores were committed but the kernel bailed — cannot safely
            // re-run on stock. This is a rare edge case (tail call after
            // stores in the same function); fall through to let the result
            // propagate (the stores are already applied, and the function
            // effectively completed its work before the tail call).
        }
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
        |(num_slots, writes_result, action, slot_map)| {
            (
                ops.as_ptr() as usize,
                num_slots,
                writes_result,
                action,
                slot_map,
            )
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

// ---------------------------------------------------------------------------
// Call runner: executes a CallInternal on behalf of the majit kernel.
// ---------------------------------------------------------------------------

/// Context for [`call_runner_fn`], capturing references from `run_jit` that
/// callee execution needs. Lives on `run_jit`'s stack frame; its raw pointer
/// is registered via [`super::majit::kernel::set_call_runner`] and valid for
/// the entire `run_persistent` duration.
#[cfg(feature = "majit-jit")]
struct CallRunnerCtx<'a> {
    store: *mut crate::store::PrunedStore,
    code: CodeView<'a>,
    instance: Inst,
    callee_stack: Stack,
}

/// The [`super::majit::kernel::CallRunnerFn`] callback. Executes a wasm
/// internal call by pushing a root frame on a separate Stack and running
/// `execute_until_done` on the stock executor.
#[cfg(feature = "majit-jit")]
fn call_runner_fn(data: *mut (), func_addr: usize, params: &[i64]) -> i64 {
    let ctx = unsafe { &mut *(data as *mut CallRunnerCtx) };
    let store = unsafe { &mut *ctx.store };

    // Recover the FuncEntry from the exposed pointer address.
    let func = unsafe {
        &*core::ptr::with_exposed_provenance::<crate::engine::code_map::FuncEntry>(func_addr)
    };

    // Compile/fetch the callee.
    let compiled = match func.get_or_compile(
        Some(store.inner_mut().fuel_mut()),
        ctx.code.features(),
    ) {
        Ok(c) => c,
        Err(_) => {
            super::majit::kernel::set_residual_trap(crate::TrapCode::UnreachableCodeReached);
            return 0;
        }
    };
    let callee_ip = Ip::from(compiled.ops());
    let len_local_slots = compiled.len_local_slots();
    let len_stack_slots = compiled.len_stack_slots();

    // Reset the callee stack for reuse.
    ctx.callee_stack.reset();

    // Push a root frame (caller_ip = None → pop returns None → done).
    let callee_params = BoundedSlotSpan::new(SlotSpan::new(Slot::from(0)), params.len() as u16);
    let callee_sp = match ctx.callee_stack.push_frame(
        None,
        callee_ip,
        callee_params,
        len_local_slots,
        len_stack_slots,
        Some(ctx.instance),
    ) {
        Ok(sp) => sp,
        Err(_) => {
            super::majit::kernel::set_residual_trap(crate::TrapCode::StackOverflow);
            return 0;
        }
    };

    // Write the staged params into the callee frame.
    for (i, &val) in params.iter().enumerate() {
        unsafe { callee_sp.set::<i64>(Slot::from(i as u16), val) };
    }

    // Execute the callee on the stock executor.
    let (mem0, mem0_len) = utils::extract_mem0(store, ctx.instance);
    let (ireg, freg32, freg64) = ctx.callee_stack.regs();
    let mut vm = VmState::new(store, &mut ctx.callee_stack, ctx.code);
    match execute_until_done(
        &mut vm, callee_ip, callee_sp,
        mem0, mem0_len, ctx.instance,
        ireg, freg32, freg64,
    ) {
        Ok(sp) => unsafe { sp.get::<i64>(Slot::from(0)) },
        Err(_) => {
            super::majit::kernel::set_residual_trap(crate::TrapCode::UnreachableCodeReached);
            0
        }
    }
}
