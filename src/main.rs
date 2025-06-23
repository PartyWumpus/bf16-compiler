use clap::Parser;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use sdl2::Sdl;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::video::Window;

use std::cell::RefCell;
use std::error::Error;
use std::process::exit;
use std::sync::LazyLock;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, mem};

const WINDOW_SIZE: u32 = 512;
const PIXEL_SCALE: u32 = WINDOW_SIZE / 16;

thread_local! {
    static PREV_TIME: RefCell<Instant> = RefCell::new(Instant::now());
    static EVENTS: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
    static SDL_CONTEXT: RefCell<Sdl> = RefCell::new(sdl2::init().unwrap());
    static SDL_CANVAS: RefCell<Canvas<Window>> = {
        let canvas = SDL_CONTEXT.with(|sdl_context| {
            let sdl_context = sdl_context.borrow();
            let video_subsystem = sdl_context.video().unwrap();

        let window = video_subsystem.window("rust-sdl2 demo", WINDOW_SIZE, WINDOW_SIZE)
        .position_centered()
        .build()
        .unwrap();

        window.into_canvas().build().unwrap()
        });

        RefCell::new(canvas)
    };
}

static ARGS: LazyLock<Args> = LazyLock::new(|| Args::parse());

fn to_opt_level(s: &str) -> Result<OptimizationLevel, String> {
    let opt_level: usize = s.parse().map_err(|_| format!("`{s}` isn't a number"))?;
    Ok(match opt_level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        3 => OptimizationLevel::Aggressive,
        4.. => return Err("Opt level not in range 0-3".to_owned()),
    })
}

#[derive(Parser, Debug)]
#[command(about="An LLVM-based brainfuck compiler", long_about = None)]
struct Args {
    /// File to compile
    filename: String,

    /// Optimization level, from 0-3
    #[arg(value_enum, long, short='O',short_alias='o', default_value = "2", value_parser=to_opt_level)]
    opt_level: OptimizationLevel,

    /// Outputs LLVM IR after each function is generated
    #[arg(short, long)]
    print_llvm_ir: bool,

    /// Display is interpreted
    #[arg(short, long)]
    grayscale: bool,
}

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
        let mut prev: Option<char> = None;
        let mut count = 0;
        for op in code.chars() {
            if prev == Some(op) {
                count += 1
            } else {
                match prev {
                    Some('>') => self.emit_inc_ptr(&pointers, count)?,
                    Some('<') => self.emit_dec_ptr(&pointers, count)?,
                    Some('+') => self.emit_inc_data(&pointers, count)?,
                    Some('-') => self.emit_dec_data(&pointers, count)?,
                    _ => (),
                }
                if matches!(op, '>' | '<' | '+' | '-') {
                    prev = Some(op);
                    count = 1;
                } else {
                    prev = None;
                };
            };
            match op {
                '[' => self.emit_loop_begin(function, &pointers)?,
                ']' => self.emit_loop_end()?,
                '.' => self.emit_call_put_fn(&pointers)?,
                ',' => self.emit_call_get_fn(&pointers)?,
                _ => (),
            }
        }

        self.builder.build_return(None).unwrap();

        if ARGS.print_llvm_ir {
            println!("{}", self.module.print_to_string().to_string());
        };

        unsafe { Ok(self.execution_engine.get_function("init")?) }
    }

    fn emit_inc_ptr(&self, pointers: &Pointers<'ctx>, count: u64) -> Result<(), Box<dyn Error>> {
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let pos = self
            .builder
            .build_int_add(pos, i16_type.const_int(count, false), "pos")?;
        self.builder.build_store(pointers.position_ptr, pos)?;
        Ok(())
    }

    fn emit_dec_ptr(&self, pointers: &Pointers<'ctx>, count: u64) -> Result<(), Box<dyn Error>> {
        let i16_type = self.context.i16_type();
        let pos: IntValue = self
            .builder
            .build_load(i16_type, pointers.position_ptr, "pos")?
            .try_into()
            .unwrap();
        let pos = self
            .builder
            .build_int_sub(pos, i16_type.const_int(count, false), "pos")?;
        self.builder.build_store(pointers.position_ptr, pos)?;
        Ok(())
    }

    fn emit_inc_data(&self, pointers: &Pointers<'ctx>, count: u64) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();

        let val_ptr = self.emit_get_val_ptr(pointers)?;
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();
        let val = self
            .builder
            .build_int_add(val, i8_type.const_int(count, false), "val")?;
        self.builder.build_store(val_ptr, val)?;
        Ok(())
    }

    fn emit_dec_data(&self, pointers: &Pointers<'ctx>, count: u64) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let val_ptr = self.emit_get_val_ptr(pointers)?;
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();
        let val = self
            .builder
            .build_int_sub(val, i8_type.const_int(count, false), "val")?;
        self.builder.build_store(val_ptr, val)?;
        Ok(())
    }

    fn emit_loop_begin(
        &mut self,
        func: FunctionValue<'ctx>,
        pointers: &Pointers<'ctx>,
    ) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();

        let begin = self.context.append_basic_block(func, "loop_begin");
        let main = self.context.append_basic_block(func, "loop_main");
        let end = self.context.append_basic_block(func, "loop_end");
        self.blocks.push((begin, end));

        self.builder.build_unconditional_branch(begin)?;
        self.builder.position_at_end(begin);

        let val_ptr = self.emit_get_val_ptr(pointers)?;
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

    fn emit_call_put_fn(&self, pointers: &Pointers<'ctx>) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let put_type = void_type.fn_type(&[i8_type.into(), ptr_type.into()], false);

        let val_ptr = self.emit_get_val_ptr(pointers)?;
        let val: IntValue = self
            .builder
            .build_load(i8_type, val_ptr, "val")?
            .try_into()
            .unwrap();
        self.builder.build_indirect_call(
            put_type,
            pointers.put_fn,
            &[val.into(), pointers.data_ptr.into()],
            "put",
        )?;
        Ok(())
    }

    fn emit_call_get_fn(&self, pointers: &Pointers<'ctx>) -> Result<(), Box<dyn Error>> {
        let i8_type = self.context.i8_type();
        let get_type = i8_type.fn_type(&[], false);
        let res: IntValue = self
            .builder
            .build_indirect_call(get_type, pointers.get_fn, &[], "get")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();

        let val_ptr = self.emit_get_val_ptr(pointers)?;
        self.builder.build_store(val_ptr, res)?;
        Ok(())
    }

    fn emit_get_val_ptr(
        &self,
        pointers: &Pointers<'ctx>,
    ) -> Result<PointerValue<'ctx>, Box<dyn Error>> {
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
        Ok(val_ptr)
    }
}

extern "C" fn put(_audio_pitch: u8, data_ptr: *mut [u8; 30000]) {
    SDL_CANVAS.with(|canvas| {
        let mut canvas = canvas.borrow_mut();
        canvas.clear();

        // TODO: play sound https://github.com/p2r3/bf16/blob/0dcbe2e6ea431a59bd9233f14fd1d3781a48b238/bf16.c#L20
        /*SDL_CONTEXT.with(|sdl_context| {
            let sdl_context = sdl_context.borrow();
            let audio_subsystem = sdl_context.audio().unwrap();
        });*/

        for y in 0..=15 {
            for x in 0..=15 {
                let pixel = unsafe { (*data_ptr)[x + y * 16] };

                let (r, g, b) = if ARGS.grayscale {
                    (pixel, pixel, pixel)
                } else {
                    // read u8 as RGB332
                    let r: u8 = (pixel & 0xE0) >> 5; // Extract top 3 bits for red
                    let g: u8 = (pixel & 0x1C) >> 2; // Extract middle 3 bits for green
                    let b: u8 = pixel & 0x03; // Extract bottom 2 bits for blue

                    // Scale the values to full 8-bit range
                    let r: u8 = ((u16::from(r) * 255) / 7) as u8;
                    let g: u8 = ((u16::from(g) * 255) / 7) as u8;
                    let b: u8 = ((u16::from(b) * 255) / 3) as u8;
                    (r, g, b)
                };

                canvas.set_draw_color(Color::RGB(r, g, b));
                canvas
                    .fill_rect(Rect::new(
                        (x as u32 * PIXEL_SCALE) as i32,
                        (y as u32 * PIXEL_SCALE) as i32,
                        PIXEL_SCALE,
                        PIXEL_SCALE,
                    ))
                    .unwrap();
            }
        }
        canvas.present();
    });

    // FIXME: this is probably memory leak
    // consider peeking sdl2's queue maybe?
    // or just cap this idk
    EVENTS.with(|events| {
        let mut event_pump = SDL_CONTEXT.with(|sdl_context| {
            let sdl_context = sdl_context.borrow();
            sdl_context.event_pump().unwrap()
        });

        let mut events = events.borrow_mut();
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => exit(0),
                _ => events.push(event),
            }
        }
    });

    PREV_TIME.with(|time| {
        let mut time = time.borrow_mut();
        sleep(Duration::from_millis(1000 / 60) - time.elapsed());
        *time = Instant::now();
    });
}

fn handle_event(event: &Event, out: &mut u8) {
    match event {
        Event::Quit { .. }
        | Event::KeyDown {
            keycode: Some(Keycode::Escape),
            ..
        } => exit(0),
        Event::KeyDown {
            keycode: Some(Keycode::Z),
            ..
        } => *out |= 0x80,
        Event::KeyDown {
            keycode: Some(Keycode::X),
            ..
        } => *out |= 0x40,
        Event::KeyDown {
            keycode: Some(Keycode::RETURN),
            ..
        } => *out |= 0x20,
        Event::KeyDown {
            keycode: Some(Keycode::SPACE),
            ..
        } => *out |= 0x10,
        Event::KeyDown {
            keycode: Some(Keycode::UP),
            ..
        } => *out |= 0x08,
        Event::KeyDown {
            keycode: Some(Keycode::DOWN),
            ..
        } => *out |= 0x04,
        Event::KeyDown {
            keycode: Some(Keycode::LEFT),
            ..
        } => *out |= 0x02,
        Event::KeyDown {
            keycode: Some(Keycode::RIGHT),
            ..
        } => *out |= 0x01,
        _ => {}
    }
}

extern "C" fn get() -> u8 {
    let mut out = 0;
    let mut event_pump = SDL_CONTEXT.with(|sdl_context| {
        let sdl_context = sdl_context.borrow();
        sdl_context.event_pump().unwrap()
    });

    let events = EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        let mut swap = vec![];
        mem::swap(&mut *events, &mut swap);
        swap
    });

    for event in events.into_iter().rev() {
        handle_event(&event, &mut out);
    }
    for event in event_pump.poll_iter() {
        handle_event(&event, &mut out);
    }

    out
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("brainfuck");
    let execution_engine = module.create_jit_execution_engine(ARGS.opt_level)?;
    let mut codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
        blocks: vec![],
    };

    let mut data_position: u16 = 0;
    let mut data = Box::new([0; 30000]);

    let init = codegen.compile_full(&fs::read_to_string(&ARGS.filename)?)?;
    let data_pos_ptr = &raw mut data_position;
    let data_ptr = &raw mut *data;

    SDL_CANVAS.with(|canvas| {
        let mut canvas = canvas.borrow_mut();
        canvas.clear();
        canvas.present();
    });

    // SAFETY: probably is not
    unsafe {
        init.call(data_pos_ptr, data_ptr, put, get);
    }

    Ok(())
}
