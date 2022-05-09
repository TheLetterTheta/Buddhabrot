#![feature(int_log)]

use clap::Parser;
use image::{imageops::rotate90, ImageBuffer, Luma};
use itertools::Itertools;
use rayon::prelude::*;
use rug::{
    float::{ParseFloatError, Round},
    ops::AssignRound,
    Complex, Float,
};
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Final resolution of the X axis (in pixels)
    #[clap(short = 'x', long, parse(try_from_str=parse_resolution), default_value = "2560x1440")]
    resolution: (u32, u32),

    /// Domain specification for the image
    /// ie. -0.5575,-0.55
    #[clap(short = 'd', long, parse(try_from_str=parse_range))]
    domain: (Float, Float),

    /// Range specification for the image
    /// ie. -0.555,-0.5525
    #[clap(short = 'r', long, parse(try_from_str=parse_range))]
    range: (Float, Float),

    /// Samples to iterate before deterimining that a
    /// point has converged
    #[clap(short = 't', long, default_value_t = 500)]
    take: usize,

    /// Output file (and format)
    #[clap(short = 'o', long)]
    output: String,

    /// Split into x images
    #[clap(short = 's', long, default_value_t = 1)]
    split: usize,
}

fn parse_resolution(resolution: &str) -> Result<(u32, u32), &'static str> {
    let (x, y) = resolution
        .split('x')
        .map(|m| m.parse::<u32>().map_err(|_| "Invalid Resolution"))
        .collect_tuple()
        .expect("Resolution must be in the format 9999x9999");
    Ok((x?, y?))
}

fn quick_check(c: &Complex) -> bool {
    const PREC: u32 = 32;
    let mut t = Complex::with_val(
        PREC,
        1.0_f32 - Complex::with_val(PREC, 1.0_f32 - Complex::with_val(PREC, 4.0_f64 * c)).sqrt(),
    );

    if t.abs().real() < &1_f32 {
        return false;
    }

    t = c - Complex::with_val(PREC, (-1_f32, 0_f32));

    // Check Period 3 bulb
    if t.abs().real() < &0.25_f32 {
        return false;
    }

    let mut creal = Float::with_val(PREC, c.real() + 1.309_f32);
    creal.square_mut();

    let mut cimag = Float::with_val(PREC, c.imag() * c.imag());
    let mut t = Float::with_val(PREC, &creal + &cimag);

    // period 4 bulb?
    if t < 0.00345_f32 {
        return false;
    }

    creal.assign_round(c.real() + 0.125_f32, Round::Down);
    creal.square_mut();
    cimag.assign_round(c.imag() - 0.744_f32, Round::Down);
    cimag.square_mut();
    t.assign_round(&creal + &cimag, Round::Down);

    // Period 3 bulb 2
    if t < 0.0088_f32 {
        return false;
    }

    cimag.assign_round(c.imag() + 0.744_f32, Round::Down);
    t.assign_round(&creal + &cimag.square(), Round::Down);
    // Period 2 bulb
    if t < 0.0088_f32 {
        return false;
    }

    true
}

fn parse_range(range: &str) -> Result<(Float, Float), ParseFloatError> {
    let (begin, end) = range.split(',').collect_tuple().unwrap();
    let precision = num_digits_log2_10(begin.len().max(end.len()));

    let begin = Float::parse(begin)?;
    let end = Float::parse(end)?;

    Ok((
        Float::with_val(precision, begin),
        Float::with_val(precision, end),
    ))
}

fn num_digits_log2_10(d: usize) -> u32 {
    let log2_10: f64 = 10_f64.log2();
    let d = d as f64 + 1_f64;
    let d = (d * log2_10).ceil().min(u32::MAX as f64);
    // Always keep 4 bits beyond the requested length
    unsafe { d.to_int_unchecked::<u32>() }
}

struct SquaresComplex {
    z: Complex,
    c: Complex,
}

impl Iterator for SquaresComplex {
    type Item = Complex;

    fn next(&mut self) -> Option<Self::Item> {
        self.z.square_mut();
        self.z += &self.c;

        let dist = Float::with_val(5, self.z.abs_ref());
        if dist > Float::with_val(5, 4_f32) {
            None
        } else {
            Some(self.z.clone())
        }
    }
}

fn square_iter(c: Complex) -> SquaresComplex {
    SquaresComplex {
        z: Complex::with_val(c.prec(), (0_f32, 0_f32)),
        c,
    }
}

fn main() {
    let args = Args::parse();

    let resolution_prec: u32 = (args.resolution.0.max(args.resolution.1).log2() + 1) as u32;

    let take = args.take;

    let (domain_start, domain_end) = args.domain;
    let (range_start, range_end) = args.range;
    let prec = resolution_prec + domain_start.prec().max(range_start.prec()) + 4;
    let x_step = Float::with_val(
        prec,
        Float::with_val(prec, &domain_end - &domain_start)
            / Float::with_val(prec, args.resolution.0),
    );
    let y_step = Float::with_val(
        prec,
        Float::with_val(prec, &range_end - &range_start) / Float::with_val(prec, args.resolution.1),
    );

    let x_begin = Float::with_val(prec, &domain_start);
    let y_begin = Float::with_val(prec, &range_start);

    let x_range = Float::with_val(prec, &domain_end - &domain_start);
    let y_range = Float::with_val(prec, &range_end - &range_start);

    println!("Bits of precision: {}", prec);

    let (mut max, mut points) = (0..args.resolution.0)
        .into_par_iter()
        .flat_map(move |x| (0..args.resolution.1).into_par_iter().map(move |y| (x, y)))
        .filter_map(|(x, y)| {
            let x_val = &x_begin + Float::with_val(prec, x * &x_step);
            let y_val = &y_begin + Float::with_val(prec, y * &y_step);

            let point = Complex::with_val(prec, (x_val, y_val));
            if !quick_check(&point) {
                return None;
            }

            let s = square_iter(point)
                .skip(1)
                .take(take)
                .collect::<Vec<Complex>>();

            if s.len() == take || s.len() < 2 {
                None
            } else {
                let t = (s.len() * args.split) / args.take;
                let s = s
                    .into_iter()
                    .filter(|p: &Complex| {
                        p.real() > &x_begin
                            && p.imag() > &y_begin
                            && p.real() < &domain_end
                            && p.imag() < &range_end
                    })
                    .map(|p| (t, p))
                    .collect::<Vec<_>>();
                Some(s)
            }
        })
        .flatten()
        .map(|(l, p): (usize, Complex)| {
            let x = Float::with_val(prec, p.real() - &x_begin) / &x_range;
            let y = Float::with_val(prec, p.imag() - &y_begin) / &y_range;

            let x = x * (args.resolution.0 - 1_u32);
            let y = y * (args.resolution.1 - 1_u32);
            (
                l,
                x.to_u32_saturating_round(Round::Nearest).expect("Got NaN"),
                y.to_u32_saturating_round(Round::Nearest).expect("Got NaN"),
            )
        })
        .fold(
            || (HashMap::new(), HashMap::new()),
            |mut m: (HashMap<usize, usize>, HashMap<(usize, u32, u32), usize>),
             p: (usize, u32, u32)| {
                let count = m.1.entry(p).or_insert(0);
                *count += 1;

                m
            },
        )
        .reduce(
            || (HashMap::new(), HashMap::new()),
            |mut m: (HashMap<usize, usize>, HashMap<(usize, u32, u32), usize>),
             mut o: (HashMap<usize, usize>, HashMap<(usize, u32, u32), usize>)| {
                for (k, v) in o.1.drain() {
                    let count = m.1.entry(k).or_insert(0);
                    let max = m.0.entry(k.0).or_insert(2);
                    *count += v;

                    if count > max {
                        *max = *count;
                    }
                }
                m
            },
        );
    let mut buckets = HashMap::new();

    for (k, v) in points.drain() {
        let i_max = max.entry(k.0).or_default();
        // now we need to scale the pixel to the max value;
        let s = (v) as f32 / (*i_max) as f32;
        let s = (s * (u8::MAX) as f32) as u8;
        let img = buckets
            .entry(k.0)
            .or_insert(ImageBuffer::new(args.resolution.0, args.resolution.1));
        img.put_pixel(k.1, k.2, Luma::<u8>([s]));
    }

    for (k, img) in buckets.drain() {
        let img = rotate90(&img);
        img.save(format!("{} - {}", k, &args.output)).ok();
        println!("Output saved to: {} - {}", k, args.output);
    }
}
