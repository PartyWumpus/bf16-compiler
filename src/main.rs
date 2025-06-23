use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{AnyValue, AsValueRef, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

use std::error::Error;
use std::ops::Add;
use std::thread::sleep;
use std::time::{Duration, Instant};

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type InitFunc = unsafe extern "C" fn(
    *mut u16,
    *mut [u8; 30000],
    extern "C" fn(u8, *mut [u8; 30000]) -> (),
    extern "C" fn() -> u8,
) -> ();

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    blocks: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
}

struct Pointers<'ctx> {
    position_ptr: PointerValue<'ctx>,
    data_ptr: PointerValue<'ctx>,
    put_fn: PointerValue<'ctx>,
    get_fn: PointerValue<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn compile_full(&mut self, code: &str) -> Result<JitFunction<'_, InitFunc>, Box<dyn Error>> {
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let fn_type = void_type.fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
            ],
            false,
        );
        let function = self.module.add_function("init", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let position_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let data_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let put_fn = function.get_nth_param(2).unwrap().into_pointer_value();
        let get_fn = function.get_nth_param(3).unwrap().into_pointer_value();

        let pointers = Pointers {
            position_ptr,
            data_ptr,
            put_fn,
            get_fn,
        };
        for op in code.chars() {
            match op {
                '>' => self.emit_inc_ptr(&pointers)?,
                '<' => self.emit_dec_ptr(&pointers)?,
                '+' => self.emit_inc_data(&pointers)?,
                '-' => self.emit_dec_data(&pointers)?,
                '[' => self.emit_loop_begin(function, &pointers)?,
                ']' => self.emit_loop_end()?,
                '.' => self.emit_put(&pointers)?,
                ',' => self.emit_get(&pointers)?,
                _ => (),
            }
        }

        self.builder.build_return(None).unwrap();

        println!("{}", self.module.print_to_string());

        unsafe { Ok(self.execution_engine.get_function("init")?) }
    }

    fn emit_inc_ptr(&self, pointers: &Pointers) -> Result<(), Box<dyn Error>> {
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let pos = self
            .builder
            .build_int_add(pos, i16_type.const_int(1, false), "pos")?;
        self.builder.build_store(pointers.position_ptr, pos)?;
        Ok(())
    }

    fn emit_dec_ptr(&self, pointers: &Pointers) -> Result<(), Box<dyn Error>> {
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let pos = self
            .builder
            .build_int_sub(pos, i16_type.const_int(1, false), "pos")?;
        self.builder.build_store(pointers.position_ptr, pos)?;
        Ok(())
    }

    fn emit_inc_data(&self, pointers: &Pointers) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let val_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(i8_type, pointers.data_ptr, &[pos], "val_ptr")?
        };
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();
        let val = self
            .builder
            .build_int_add(val, i8_type.const_int(1, false), "val")?;
        self.builder.build_store(val_ptr, val)?;
        Ok(())
    }

    fn emit_dec_data(&self, pointers: &Pointers) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let val_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(i8_type, pointers.data_ptr, &[pos], "val_ptr")?
        };
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();
        let val = self
            .builder
            .build_int_sub(val, i8_type.const_int(1, false), "val")?;
        self.builder.build_store(val_ptr, val)?;
        Ok(())
    }

    fn emit_loop_begin(
        &mut self,
        func: FunctionValue<'ctx>,
        pointers: &Pointers,
    ) -> Result<(), Box<dyn Error>> {
        let begin = self.context.append_basic_block(func, "loop_begin");
        let main = self.context.append_basic_block(func, "loop_main");
        let end = self.context.append_basic_block(func, "loop_end");
        self.blocks.push((begin, end));

        self.builder.build_unconditional_branch(begin)?;
        self.builder.position_at_end(begin);

        let i8_type = self.context.i8_type();
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let val_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(i8_type, pointers.data_ptr, &[pos], "val_ptr")?
        };
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();

        let iszero = self.builder.build_int_compare(
            IntPredicate::EQ,
            val,
            i8_type.const_zero(),
            "iszero",
        )?;
        self.builder.build_conditional_branch(iszero, end, main)?;
        self.builder.position_at_end(main);

        Ok(())
    }

    fn emit_loop_end(&mut self) -> Result<(), Box<dyn Error>> {
        let (start, end) = self.blocks.pop().unwrap();
        self.builder.build_unconditional_branch(start)?;
        self.builder.position_at_end(end);
        Ok(())
    }

    fn emit_put(&self, pointers: &Pointers) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let i16_type = self.context.i16_type();
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let put_type = void_type.fn_type(&[i8_type.into(), ptr_type.into()], false);
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let val_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(i8_type, pointers.data_ptr, &[pos], "val_ptr")?
        };
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();
        self.builder
            .build_indirect_call(put_type, pointers.put_fn, &[val.into(), pointers.data_ptr.into()], "wawa")?;
        Ok(())
    }

    fn emit_get(&self, pointers: &Pointers) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let i16_type = self.context.i16_type();
        let get_type = i8_type.fn_type(&[], false);
        let res: IntValue = self
            .builder
            .build_indirect_call(get_type, pointers.get_fn, &[], "wawa")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();

        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let val_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(i8_type, pointers.data_ptr, &[pos], "val_ptr")?
        };
        self.builder.build_store(val_ptr, res)?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;
    let mut codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
        blocks: vec![],
    };

    let mut data_position: u16 = 0;
    let mut data = Box::new([0; 30000]);
    println!("{:?}", data_position);
    
    let str = r##"# program memory map (after framebuffer):
#   0:  memory start marker for gliders
#   1:  player position/hole
#   2:  player position copy (near)
#   3:  player position copy (far)
#   4:  input source
#   5:  temp input flag
#   6:  input copy (near)
#   7:  input copy (far; cleared immediately)
#   8:  input update cycle
#   9:  input update cycle copy (near)
#   10: input update cycle temp flag
#   11: rng timer
#   12: apple position
#   13: apple position copy (near)
#   14: apple position copy (far)
#   15: anchor point (always 0)
#   16: anchor point (always 1)
#   17: apple position copy/tmp flag
#   18: score
#   19: score copy (near)
#   20: score copy temp flag
#   21: diagonal movement flag
#   22: diagonal movement flag copy (near)
#   23: diagonal movement flag copy (far)/bad apple position flag
#   24: bad apple position
#   25: bad apple position copy (near)
#   26: player tail offset/bad apple position copy (far)
#   27: anchor point (always 1)
#   28: player tail copy (far)
#   29: player tail copy (near)
#   30: player tail (from this address onward)
#   47: sound

# move to program ram
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# leave marker at start of program ram
+
# initial player position
>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # 120
# initial player movement
>>>++++++++++++++++
# initial apple positions
>>>>>>>>++++++++++++++++++++++++++ # 26 (good apple)
>>>>>>>>>>>>++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # 200 (bad apple)
# set up anchor points
<<<<<<<<+>>>>>>>>>>>+
# return to memory start marker
<<<<<<<<<<<<<<<<<<<<<<<<<<<

# main program loop
[
  # ========================================================================================
  #                                     PLAYER RENDERING
  # ========================================================================================
  # spread new player position
  # this notably leaves a hole where the position used to be
  >[->+>+<<]
  # go to first position copy
  >
  # mark framebuffer up to player position address
  [
    -<<<      # take one from position; go to framebuffer
    [<]       # glide over to the first zero value
    +         # add to it making it nonzero
    <[+<]>    # add one to next value if it's positive
              # ^ this counteracts changing color when clearing marks
    [>]       # glide back to the position hole
    >         # step back into first position copy
  ]
  # framebuffer has been marked; position copy is empty
  # clean up marks and draw a green pixel at the final position
  <<<[<]
    ++++++++++++++++++++ # 52 (00011000) minus 32 = 20
  >[->]
  # we're back at the hole; move to program ram marker and reset it
  <+
  # move last player position copy back into place
  >>>[-<<+>>]<<< # this puts us back at the ram start marker

  # ========================================================================================
  #                                     APPLE COLLISION
  # ========================================================================================
  >[->+>+<<] # spread player position again; we'll need to use it for a comparison
  >>>>>>>>>>> # go to apple position
  [->+>>>>+<<<<<]>>>>>[-<<<<<+>>>>>] # copy apple position one address to the right
  + # set temporary flag
  <<<<<<<<<<<<<<< # go to near player position copy (address 2)
  # subtract player position from apple position (address 13)
  -[->>>>>>>>>>>-<<<<<<<<<<<]
  >[-<+<+>>]< # copy and restore player position
  >>>>>>>>>>> # return to apple position copy (address 13)
  # if resulting value is exactly 0: player is touching it
  [>>>>-<<<<[-]] # if value is NOT 0 this will clear the flag and value
  # check if flag is set; if so: update apple position
  >>>>[-
    <<<<<<    # go to rng timer address
    [->+<]    # copy it on top of apple position
    >>>>>>>   # go to score address
    +         # increment score by 1
    >>>>>>[-] # clear bad apple position
    <<<<<<<<<<<<<<<<<<<<<< # go to near player position (recopied earlier)
    # move bad apple 100 pixels ahead of the player
    [->>>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # play sound effect
    >>>>>>>>>>>>>>>>>>>>>>>+++++++++++++++++++++++++++++++++++++++++++++++++++++++++<<<<<<<<<<<<<<<<<<<<<<<
    # return to flag address
    <<<<<<<
  ]
  <<<<<<<<<<<<<<<[-] # ensure near player position is cleared
  << # return to memory marker address

  # ========================================================================================
  #                                     APPLE RENDERING
  # ========================================================================================
  >[->+>+<<] # spread player position again; we'll need to use it for a comparison
  >>>>>>>>>>> # go to apple position
  [->+>+>>>+<<<<<]>>>>>[-<<<<<+>>>>>] # copy apple position two addresses to the right
  + # set temporary flag
  <<<<<<<<<<<<<<< # go to near player position copy (address 2)
  # compare player position to apple position (address 14)
  [-[
    -             # subtract from player position
    >>>>>>>>>>>>  # go to apple position
    [->]          # do a "soft subtraction": cap min value at zero
    >[<]          # to normalize the address we attach to and glide from known anchor points
    <<<<<<<<<<<<< # return to player position
  ]]
  >>>>>>>>>>>> # return to apple position copy (address 14)
  # if resulting value is 0: player is behind it
  [>>>-<<<[-]] # if value is NOT 0 this will clear the flag and value
  >>>[-<<<<+>>>>] # check if flag is set; increment position copy if apple is in front
  # render the apple: same procedure as player
  <<<< # return to near (adjusted) apple position copy
  [
    -<<<<<<<<<<<<<< # take one from position; go to framebuffer
    [<]             # glide over to the first zero value
    +               # add to it making it nonzero
    <[+<]>          # add one to next value if it's positive
                    # ^ this counteracts changing color when clearing marks
    [>]             # glide to player position hole
    >>>>>>>>>>>>    # step back into first apple position copy
  ]

  <<<<<<<<<< # go to player position copy (far)
  [-<<+>>] # restore player position value; go to far copy address
  + # make far copy address positive (to use as anchor below)
  <<<-< # turn memory marker into a hole; go to framebuffer
  # edge case: fix potential discoloration of first pixel
  # this checks if the first pixel is 1 (meaning it was 0) and increments it
  # the marker removal process later would otherwise decrement this too far
  # for other pixels: the main object drawing loop already covers this
  -[+>]>>>[<]<<<+>>>>-<<<<

  # clean up marks and draw red pixel at the final position
  [<] # glide to end of framebuffer marks
  ---------------------------------------------------------------- # 224 (11100000) minus 32 = 192 (neg 64)
  >[->] # glide back to player position hole and clean up marks on the way
  + # fix program ram marker

  # ========================================================================================
  #                                   BAD APPLE COLLISION
  # ========================================================================================
  >[->+>+<<] # spread player position again; we'll need to use it for a comparison
  >>>>>>>>>>>>>>>>>>>>>>> # go to bad apple position
  [->+>+<<]>>[-<<+>>] # copy apple position one address to the right
  + # set temporary flag
  <<<<<<<<<<<<<<<<<<<<<<<< # go to near player position copy (address 2)
  # subtract player position from apple position (address 25)
  --[->>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<]
  >>>>>>>>>>>>>>>>>>>>>>> # return to apple position copy (address 25)
  # if resulting value is exactly 0: player is touching it
  [>-<[-]] # if value is NOT 0 this will clear the flag and value
  # check if flag is set; if so: die
  >[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]
  <<<<<<<<<<<<<<<<<<<<<<<<[-] # ensure near player position is clear
  >[-<<+>>]<<< # restore player position from far position copy

  # ========================================================================================
  #                                   BAD APPLE RENDERING
  # ========================================================================================
  >[->+>+<<] # spread player position again; we'll need to use it for a comparison
  >>>>>>>>>>>>>>>>>>>>>>> # go to apple position
  [-<+>>+>+<<]<[->+<] # copy apple position two addresses to the right
  + # set temporary flag
  # janky setup of some anchors: 27 is 0; 28 and 29 are 1
  >>>>->+>+
  # this is the same routine as below loop but with a soft *increment*
  # refer to that code for a more in depth explanation of the idea
  # this just fixes a graphical bug caused by a strange off by one error
  <<<[+>]>[<]<<<<<<<<<<<<<<<<<<<<<<<<<
  # compare player position to apple position (address 26)
  [-[
    -                         # subtract from player position
    >>>>>>>>>>>>>>>>>>>>>>>>  # go to apple position
    [->]                      # do a "soft subtraction": cap min value at zero
    >[<]                      # to normalize the address we attach to and glide from known anchor points
    <<<<<<<<<<<<<<<<<<<<<<<<< # return to player position
  ]]
  >>>>>>>>>>>>>>>>>>>>>>>> # return to apple position copy (address 26)
  # if resulting value is 0: player is behind it
  [<<<->>>[-]] # if value is NOT 0 this will clear the flag and value
  <[->+>+<<]>>[-<<+>>]<< # veeeery hacky way of recopying bad apple position to far
  # ^ this only works because anchor at address 27 is temporarily 0
  <<[->>+<<] # check if flag is set; increment position copy if apple is in front

  + # reset flag
  # copy GOOD apple position (to near); select it
  <<<<<<<<<<<[->+>+<<]>>[-<<+>>]<
  # compare GOOD apple position to BAD apple position
  [
    -              # subtract from good apple position
    >>>>>>>>>>>>>  # go to bad apple position (far)
    [->]           # do a "soft subtraction": cap min value at zero
    >[<]           # normalize address to 27: glide from known anchor points
    <<<<<<<<<<<<<< # return to good apple position
  ]
  >>>>>>>>>>>>> # return to bad apple position copy (address 26)
  [<<<->>>[-]] # if value is NOT 0 this will clear the flag and value
  <<<[->>+<<] # check if flag is set; increment position copy if needed

  >>>>+>->-<<<<<< # undo janky flag setup

  # render the apple: same procedure as player
  >> # return to near (adjusted) apple position copy
  [
    -<<<<<<<<<<<<<<<<<<<<<<<<<< # take one from position; go to framebuffer
    [<]             # glide over to the first zero value
    +               # add to it making it nonzero
    <[+<]>          # add one to next value if it's positive
                    # ^ this counteracts changing color when clearing marks
    [>]             # glide to player position hole
    >>>>>>>>>>>>>>>>>>>>>>>> # step back into first apple position copy
  ]

  <<<<<<<<<<<<<<<<<<<<<< # go to player position copy (far)
  [-<<+>>] # restore player position value; go to far copy address
  + # make far copy address positive (to use as anchor below)
  <<<-< # turn memory marker into a hole; go to framebuffer
  # edge case: fix potential discoloration of first pixel
  # this checks if the first pixel is 1 (meaning it was 0) and increments it
  # the marker removal process later would otherwise decrement this too far
  # for other pixels: the main object drawing loop already covers this
  -[+>]>>>[<]<<<+>>>>-<<<<

  # clean up marks and draw cyan pixel at the final position
  [<] # glide to end of framebuffer marks
  ----- # 27 (00011011) minus 32 = 251 (neg 5)
  >[->] # glide back to player position hole and clean up marks on the way
  + # fix program ram marker

  # ========================================================================================
  #                                       TAIL HANDLER
  # ========================================================================================
  >>>>>>>>>>>>>>>>>> # go to score address
  [->+>+<<]>>[-<<+>>]< # copy it one cell to the right; enter that cell
  [[ # score is at least 1:
    [-]                   # clear score copy
    >>>>>>>>>>>           # go to tail position address 1
    [-<+<+>>]<<[->>+<<]   # copy it to the left; select far copy

    <<<<<<<<<<<<<<<<<<<<<<<<<<< # go to player position
    [->+>+<<]> # spread player position; go to its near copy
    # compare player position (near) to tail piece position (near)
    -[
      -                           # subtract from player position
      >>>>>>>>>>>>>>>>>>>>>>>>>>> # go to tail position
      [-<]                        # do a "soft subtraction": cap min value at zero
      <[>]                        # normalize to address 28 with anchor points
      <<<<<<<<<<<<<<<<<<<<<<<<<<  # return to player position
    ]
    >>>>>>>>>>>>>>>>>>>>>>>>>>> # return to near tail position copy (address 29)
    # if resulting value is 0: player is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >[-<+<+>>]<<[->>+<<]> # recopy tail position; select near copy

    <<<<<<<<<<<<<<<<< # go to apple position
    [->+>+<<]>>[-<<+>>]< # copy apple position (to near); select it
    # compare apple position (near) to tail piece position (near)
    [
      -                           # subtract from apple position
      >>>>>>>>>>>>>>>>            # go to tail position
      [-<]                        # do a "soft subtraction": cap min value at zero
      <[>]                        # normalize to address 28 with anchor points
      <<<<<<<<<<<<<<<             # return to apple position
    ]

    >>>>>>>>>>>>>>>> # return to near tail position copy (address 29)
    # if resulting value is 0: apple is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >[-<+<+>>]<<[->>+<<]> # recopy tail position; select near copy

    <<<<< # go to BAD apple position
    # copy BAD apple position to GOOD apple near; select it
    # we're hijacking the anchors that the good apple structure provides
    [->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<
    # compare apple position (near) to tail piece position (near)
    [
      -                           # subtract from apple position
      >>>>>>>>>>>>>>>>            # go to tail position
      [-<]                        # do a "soft subtraction": cap min value at zero
      <[>]                        # normalize to address 28 with anchor points
      <<<<<<<<<<<<<<<             # return to apple position
    ]

    >>>>>>>>>>>>>>>> # return to near tail position copy (address 29)
    # if resulting value is 0: apple is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >[-<+<+>>]<<[->>+<<]> # recopy tail position; select near copy

    <<<[->>>+<<<]>>> # apply tail offset

    # draw the tail piece
    [
      # take one from position; go to framebuffer
      -<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      [<]             # glide over to the first zero value
      +               # add to it making it nonzero
      <[+<]>          # add one to next value if it's positive
                      # ^ this counteracts changing color when clearing marks
      [>]             # glide to player position hole
      # step back into tailpiece position copy
      >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ]
    # clean up marks and draw dark green pixel at the final position
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<] # glide to end of framebuffer marks
    ++++++++++++ # 44 (00010000) minus 32 = 12
    >[->] # glide back to player position hole and clean up marks on the way

    <+ # fix program ram marker
    >>>[-<<+>>]<<< # restore player position value
    >>>>>>>>>>>>>>>>>>> # return to score copy (near)
  ]]

  < # go to score address
  [->+>+<<]>>[-<<+>>]< # copy it one cell to the right; enter that cell
  [-[ # score is at least 2:
    [-]                                # clear score copy
    >>>>>>>>>>>(>)                     # go to tail position address 2
    [-(<)<+<+>>(>)](<)<<[->>(>)+<<(<)] # copy it to the left; select far copy

    <<<<<<<<<<<<<<<<<<<<<<<<<<< # go to player position address
    [->+>+<<]>>[-<<+>>]< # copy player position one to the right
    [->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<] # subtract from tail position
    >>>>>>>>>>>>>>>>>>>>>>>>>> # go to FAR tail position copy address
    +> # set temporary flag
    [[-]<->] # if remaining position is NOT 0: clear flag
    # if flag was not cleared: player has died; stall the game
    <[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]
    >>(>)[-(<)<+<+>>(>)](<)<<[->>(>)+<<(<)] # recopy tail position
    <<<<<<<<<<<<<<<<<<<<<<<<<<< # return to player position address

    [->+>+<<]> # spread player position; go to its near copy
    # compare player position (near) to tail piece position (near)
    -[
      -                           # subtract from player position
      >>>>>>>>>>>>>>>>>>>>>>>>>>> # go to tail position
      [-<]                        # do a "soft subtraction": cap min value at zero
      <[>]                        # normalize to address 28 with anchor points
      <<<<<<<<<<<<<<<<<<<<<<<<<<  # return to player position
    ]
    >>>>>>>>>>>>>>>>>>>>>>>>>>> # return to near tail position copy (address 29)
    # if resulting value is 0: player is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >(>)[-(<)<+<+>>(>)](<)<<[->>(>)+<<(<)]> # recopy tail position; select near copy

    <<<<<<<<<<<<<<<<< # go to apple position
    [->+>+<<]>>[-<<+>>]< # copy apple position (to near); select it
    # compare apple position (near) to tail piece position (near)
    [
      -                           # subtract from apple position
      >>>>>>>>>>>>>>>>            # go to tail position
      [-<]                        # do a "soft subtraction": cap min value at zero
      <[>]                        # normalize to address 28 with anchor points
      <<<<<<<<<<<<<<<             # return to apple position
    ]

    >>>>>>>>>>>>>>>> # return to near tail position copy (address 29)
    # if resulting value is 0: apple is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >(>)[-(<)<+<+>>(>)](<)<<[->>(>)+<<(<)]> # recopy tail position; select near copy

    <<<<< # go to BAD apple position
    # copy BAD apple position to GOOD apple near; select it
    # we're hijacking the anchors that the good apple structure provides
    [->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<
    # compare apple position (near) to tail piece position (near)
    [
      -                           # subtract from apple position
      >>>>>>>>>>>>>>>>            # go to tail position
      [-<]                        # do a "soft subtraction": cap min value at zero
      <[>]                        # normalize to address 28 with anchor points
      <<<<<<<<<<<<<<<             # return to apple position
    ]

    >>>>>>>>>>>>>>>> # return to near tail position copy (address 29)
    # if resulting value is 0: apple is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >(>)[-(<)<+<+>>(>)](<)<<[->>(>)+<<(<)]> # recopy tail position; select near copy

    # copy previous tail piece position into score copy addresses
    (>)[-(<)<<<<<<<<<+<+>>>>>>>>>>(>)](<)<<<<<<<<<[->>>>>>>>>(>)+<<<<<<<<<(<)]<
    # ^ that leaves us at near score copy (address 19: contains previous tail piece)
    # compare previous tail position to current tail position
    -[
      -          # subtract from previous tail position
      >>>>>>>>>> # go to current tail position copy
      [-<]       # do a "soft subtraction": cap min value at zero
      <[>]       # normalize to address 28 with anchor points
      <<<<<<<<<  # return to previous tail position
    ]
    >>>>>>>>>> # return to current tail position copy (address 29)
    # if resulting value is 0: previous tail piece is behind it
    [<<<->>>[-]] # if value is NOT 0: decrement player tail offset
    >(>)[-(<)<+<+>>(>)](<)<<[->>(>)+<<(<)]> # recopy tail position; select near copy

    <<<[->>>+<<<]>>> # apply tail offset

    # draw the tail piece
    [
      # take one from position; go to framebuffer
      -<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      [<]             # glide over to the first zero value
      +               # add to it making it nonzero
      <[+<]>          # add one to next value if it's positive
                      # ^ this counteracts changing color when clearing marks
      [>]             # glide to player position hole
      # step back into tailpiece position copy
      >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ]
    # clean up marks and draw dark green pixel at the final position
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<] # glide to end of framebuffer marks
    ++++++++++++ # 44 (00010000) minus 32 = 12
    >[->] # glide back to player position hole and clean up marks on the way

    <+ # fix program ram marker
    >>>[-<<+>>]<<< # restore player position value
    >>>>>>>>>>>>>>>>>>> # return to score copy (near)
  ]]

  # this large block of code is generated using tailgen(dot)js
  # it's conceptually the same as the second tail piece code above
  #   but with the parts in brackets multiplied based on tail length
  <[->+>+<<]>>[-<<+>>]<[-[-[[-]>>>>>>>>>>>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>[-<<<+<+>>>>]<<<<[->>>>+<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[[-]>>>>>>>>>>>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>[-<<<<+<+>>>>>]<<<<<[->>>>>+<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[[-]>>>>>>>>>>>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>[-<<<<<+<+>>>>>>]<<<<<<[->>>>>>+<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>[-<<<<<<+<+>>>>>>>]<<<<<<<[->>>>>>>+<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>[-<<<<<<<+<+>>>>>>>>]<<<<<<<<[->>>>>>>>+<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>[-<<<<<<<<+<+>>>>>>>>>]<<<<<<<<<[->>>>>>>>>+<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>[-<<<<<<<<<+<+>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]]]]]]
  <[->+>+<<]>>[-<<+>>]<[-[-[-[-[-[-[-[-[-[-[-[-[-[-[[-]>>>>>>>>>>>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>+>[[-]<->]<[+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++..........---..........----..............................--------------------------------------------------------[-.+]]>>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<<<<<<<<<<<<[->+>+<<]>-[->>>>>>>>>>>>>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<<<<<<<<<<<<<<<<[->+>+<<]>>[-<<+>>]<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<<<<[->+<<<<<<<<<<<<+>>>>>>>>>>>]>[-<+>]<<<<<<<<<<<<[->>>>>>>>>>>>>>>>[-<]<[>]<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>[-<<<<<<<<<<+<+>>>>>>>>>>>]<<<<<<<<<<[->>>>>>>>>>+<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>[-<<<<<<<<<<<+<+>>>>>>>>>>>>]<<<<<<<<<<<[->>>>>>>>>>>+<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>[-<<<<<<<<<<<<+<+>>>>>>>>>>>>>]<<<<<<<<<<<<[->>>>>>>>>>>>+<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>[-<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>]<<<<<<<<<<<<<[->>>>>>>>>>>>>+<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>[-<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<[->>>>>>>>>>>>>>+<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>[-<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>[-<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<<<]<-[->>>>>>>>>>[-<]<[>]<<<<<<<<<]>>>>>>>>>>[<<<->>>[-]]>>>>>>>>>>>>>>>[-<<<<<<<<<<<<<<<+<+>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<[->>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<]<<[->>>+<<<]>>>[-<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]+<[+<]>[>]>>>>>>>>>>>>>>>>>>>>>>>>>>>>]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[<]++++++++++++>[->]<+>>>[-<<+>>]>>>>>>>>>>>>>>>>]]]]]]]]]]]]]]]

  <<<<<<<<<<<<<<<<<<< # return to ram marker

  # ========================================================================================
  #                                      INPUT HANDLING
  # ========================================================================================
  # read input; apply changes only if new input not empty or invalid
  >>>>>, # read new input into input temp flag address
  [->+>+<<] # spread value across input copy cells
  > # go to near input copy cell
  [ # if not zero (no input)
  ---[ # and not 3 (LEFT;RIGHT)
  ----[ # and not 7 (LEFT;RIGHT;DOWN)
    ++++++ # and less than 11 (end of valid input range)
    [-<]<[>]> # * the soft subtraction routine here is actually pretty interesting
    [-<]<[>]> # * in order for this to work we typically need a few anchor points
    [-<]<[>]> # * in this case: the memory is too tightly packed to create anchors
    [-<]<[>]> # * instead we find memory that behaves like an anchor
    [-<]<[>]> # * address 5 (tmp input flag) is cleared above: that's our 0 anchor
    [-<]<[>]> # * address 4 (input source) can never be 0: this very piece of
    [-<]<[>]> #   code prevents it from doing so; though i did have to initialize
    [-<]<[>]> #   player movement to 16 (SPACE) to make it start nonzero as well
    [-<]<[>]>
    >>>+<<< # use address 9 (cycler near copy) as a flag for reversing condition
    [>>>-<<<[-]] # if comparison left nonzero: clear flag and near input copy
    >>>[- # check flag and reset it
      <<<<<[-] # allow input change: clear previous frame's input
      >>> # go to far input copy cell
      [-<<<+>>>] # move its value to input source address
      >> # exit loop at address 9 flag
    ]<<< # return to near input copy cell
  ]]]
  >[-] # ensure far input copy cell is clear
  > # go to input update cycle address

  # add 32: overflows every 8th frame
  ++++++++++++++++++++++++++++++++
  [->+>+<<]>>[-<<+>>]< # copy cycle value 1 cell to the right; go to address of copy
  >+< # set temporary flag to the right
  [>-<[-]] # if value is NOT 0 this will clear the flag and value
  # check if flag is set; process input if so
  >[-

    >+++++++++++++++++++++ # update rng timer

    # flip diagonal movement flag: this controls the direction of diagonal inputs
    # flipping this every time input is handled makes the movement stepped
    >>>>>>>>>>
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # shift tail position values to the right
    >>>>>>>>> # go to start of tail position block
    >>>>>>>>>>>>>>> # go to last tail address
    # shift cells one to the right and jump left
    [->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<[->+<]<
    <<<<<<<<<<<<<<<<<<<<<<<<<<<< # go to player position address
    [->+>+<<]>>[-<<+>>]< # copy position one cell to the right; enter that cell
    [->>>>>>>>>>>>>>>>>>>>>>>>>>>>+<<<<<<<<<<<<<<<<<<<<<<<<<<<<] # move position copy to tail

    >> # go to player input source address

    # update player position if input is 1 (RIGHT)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      -[>-<[-]]        # if input is NOT 1 this will clear the flag and the input
      >[-<<<<->>>>]<   # check if the flag is still set; change position if so
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 2 (LEFT)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      --[>-<[-]]       # if input is NOT 2 this will clear the flag and the input
      >[-<<<<+>>>>]<   # check if the flag is still set; change position if so
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 4 (DOWN)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      ----[>-<[-]]     # if input is NOT 4 this will clear the flag and the input
      >[-<<<<---------------->>>>]< # check if the flag is still set; change position if so
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 8 (UP)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      --------[>-<[-]] # if input is NOT 8 this will clear the flag and the input
      >[-<<<<++++++++++++++++>>>>]< # check if the flag is still set; change position if so
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 1 plus 4 (RIGHT;DOWN)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      -----[>-<[-]]    # if input is NOT 5 this will clear the flag and the input
      >[- # check if the flag is still set; change position if so
        # handle diagonal movement via diagonal movement flag
        # spread flag value; process zero flag value
        >>>>>>>>>>>>>>>>[->+>+<<]>
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        [<<<<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>>>>[-]]
        # spread flag value; process nonzero flag value
        >[-<+<+>>]<
        [<<<<<<<<<<<<<<<<<<<<<---------------->>>>>>>>>>>>>>>>>>>>>[-]]
        # return to temp input flag
        <<<<<<<<<<<<<<<<<
      ]<
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 2 plus 4 (LEFT;DOWN)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      ------[>-<[-]]   # if input is NOT 6 this will clear the flag and the input
      >[- # check if the flag is still set; change position if so
        # handle diagonal movement via diagonal movement flag
        # spread flag value; process zero flag value
        >>>>>>>>>>>>>>>>[->+>+<<]>
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        [<<<<<<<<<<<<<<<<<<<<<+>>>>>>>>>>>>>>>>>>>>>[-]]
        # spread flag value; process nonzero flag value
        >[-<+<+>>]<
        [<<<<<<<<<<<<<<<<<<<<<---------------->>>>>>>>>>>>>>>>>>>>>[-]]
        # return to temp input flag
        <<<<<<<<<<<<<<<<<
      ]<
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 1 plus 8 (RIGHT;UP)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      ---------[>-<[-]] # if input is NOT 9 this will clear the flag and the input
      >[- # check if the flag is still set; change position if so
        # handle diagonal movement via diagonal movement flag
        # spread flag value; process zero flag value
        >>>>>>>>>>>>>>>>[->+>+<<]>
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        [<<<<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>>>>[-]]
        # spread flag value; process nonzero flag value
        >[-<+<+>>]<
        [<<<<<<<<<<<<<<<<<<<<<++++++++++++++++>>>>>>>>>>>>>>>>>>>>>[-]]
        # return to temp input flag
        <<<<<<<<<<<<<<<<<
      ]<
      >>[-<<+>>]<<     # restore copied position
    # update player position if input is 2 plus 8 (LEFT;UP)
      [->>+>+<<<]      # spread input value 2 cells to the right
      >>>[-<<<+>>>]<<< # move rightmost spread to original address
      >+<              # set temporary flag to the right
      ----------[>-<[-]] # if input is NOT 10 this will clear the flag and the input
      >[- # check if the flag is still set; change position if so
        # handle diagonal movement via diagonal movement flag
        # spread flag value; process zero flag value
        >>>>>>>>>>>>>>>>[->+>+<<]>
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        [<<<<<<<<<<<<<<<<<<<<<+>>>>>>>>>>>>>>>>>>>>>[-]]
        # spread flag value; process nonzero flag value
        >[-<+<+>>]<
        [<<<<<<<<<<<<<<<<<<<<<++++++++++++++++>>>>>>>>>>>>>>>>>>>>>[-]]
        # return to temp input flag
        <<<<<<<<<<<<<<<<<
      ]<
      >>[-<<+>>]<<     # restore copied position

    >>>>>> # return to input cycler flag address
  ]

  # return to program ram marker
  <<<<<<<<<<

  # ========================================================================================
  #                              FRAMEBUFFER REFRESH AND SOUND
  # ========================================================================================
  # apply background color
  <++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++
  <++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++<++++++++++++++++++++++++++++++++
  # return to start of program memory
  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  # go to sound address; render frame and play sound; return to start of program memory
  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.[-]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  # go to start of framebuffer
  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  # clear every cell and return to program memory
  [-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>
  [-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>[-]>
]"##;
  

    let sum = codegen.compile_full(str)?;
    let data_pos_ptr = &raw mut data_position;
    let data_ptr = &raw mut *data;

    extern "C" fn put(x: u8, data_ptr: *mut [u8; 30000]) {
        print!("\x1b[?25l"); // hide cursor
        println!("playing {:?}", x);
        for y in 0..=15 {
            for x in 0..=15 {
                print!("\x1b[38;5;{}m█", unsafe{(*data_ptr)[x + y*16]})
            }
            print!("\n");
        }
        print!("\x1b[17A"); // move cursor up
        print!("\x1b[?25h"); // show cursor
        print!("\x1b[0m"); // reset
        sleep(Duration::from_millis(1000/60));

        //println!("{}", x as char);
    }

    extern "C" fn get() -> u8 {
        return 0x01;
    }

    unsafe {
        sum.call(data_pos_ptr, data_ptr, put, get);
    }

    Ok(())
}
