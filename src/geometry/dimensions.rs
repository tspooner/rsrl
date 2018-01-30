//! Dimension representations module.

use super::span::Span;

use rand::ThreadRng;
use rand::distributions::{Range as RngRange, IndependentSample};

use serde::{Deserialize, Deserializer, de};
use serde::de::Visitor;
use std::{cmp, f64, fmt};
use std::fmt::Debug;
use std::ops::Range;


/// The basic dimension type.
pub trait Dimension {
    /// The corresponding primitive type.
    type Value: Debug + Clone;

    /// Sample a random value contained by this dimension.
    fn sample(&self, rng: &mut ThreadRng) -> Self::Value;

    /// Map a compatible input into a valid value of this dimension.
    fn convert(&self, val: f64) -> Self::Value;

    /// Returns the total span of this dimension.
    fn span(&self) -> Span;
}

/// Dimension type with saturating upper/lower bounds.
pub trait BoundedDimension: Dimension
    where Self::Value: PartialOrd
{
    /// The upper/lower bound type; not necessarily equal to `Dimension::Value`.
    type ValueBound: PartialOrd + Copy;

    /// Returns a reference to the dimension's lower value bound (inclusive).
    fn lb(&self) -> &Self::ValueBound;

    /// Returns a reference to the dimension's upper value bound (exclusive).
    fn ub(&self) -> &Self::ValueBound;

    /// Returns an owned tuple of the lower and upper bounds on the dimension.
    fn limits(&self) -> (Self::ValueBound, Self::ValueBound) {
        (*self.lb(), *self.ub())
    }

    /// Returns true iff `val` is within the dimension's bounds.
    fn contains(&self, val: Self::ValueBound) -> bool;
}

/// Dimension type with bounds and a finite set of values.
pub trait FiniteDimension: BoundedDimension
    where Self::Value: PartialOrd
{
    /// Returns the finite range of values in this dimension.
    fn range(&self) -> Range<Self::Value>;
}


/// A null dimension.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct Null;

impl Dimension for Null {
    type Value = ();

    fn sample(&self, _: &mut ThreadRng) -> () {
        ()
    }

    fn convert(&self, _: f64) -> Self::Value {
        ()
    }

    fn span(&self) -> Span {
        Span::Null
    }
}


/// An infinite dimension.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct Infinite;

impl Infinite {
    pub fn bounded(self, lb: f64, ub: f64) -> Continuous {
        Continuous::new(lb, ub)
    }
}

impl Dimension for Infinite {
    type Value = f64;

    fn sample(&self, _: &mut ThreadRng) -> f64 {
        unimplemented!()
    }

    fn convert(&self, val: f64) -> Self::Value {
        val
    }

    fn span(&self) -> Span {
        Span::Infinite
    }
}


/// A continous dimension.
#[derive(Clone, Copy, Serialize)]
pub struct Continuous {
    lb: f64,
    ub: f64,

    #[serde(skip_serializing)]
    range: RngRange<f64>,
}

impl Continuous {
    pub fn new(lb: f64, ub: f64) -> Continuous {
        Continuous {
            lb: lb,
            ub: ub,
            range: RngRange::new(lb, ub),
        }
    }
}

impl Dimension for Continuous {
    type Value = f64;

    fn sample(&self, rng: &mut ThreadRng) -> f64 {
        self.range.ind_sample(rng)
    }

    fn convert(&self, val: f64) -> Self::Value {
        clip!(self.lb, val, self.ub)
    }

    fn span(&self) -> Span {
        Span::Infinite
    }
}

impl BoundedDimension for Continuous {
    type ValueBound = Self::Value;

    fn lb(&self) -> &f64 {
        &self.lb
    }

    fn ub(&self) -> &f64 {
        &self.ub
    }

    fn contains(&self, val: Self::ValueBound) -> bool {
        (val >= self.lb) && (val < self.ub)
    }
}

impl<'de> Deserialize<'de> for Continuous {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        enum Field {
            Lb,
            Ub,
        };
        const FIELDS: &'static [&'static str] = &["lb", "ub"];

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`lb` or `ub`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                        where E: de::Error
                    {
                        match value {
                            "lb" => Ok(Field::Lb),
                            "ub" => Ok(Field::Ub),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct ContinuousVisitor;

        impl<'de> Visitor<'de> for ContinuousVisitor {
            type Value = Continuous;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Continuous")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Continuous, V::Error>
                where V: de::SeqAccess<'de>
            {
                let lb = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let ub = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Continuous::new(lb, ub))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Continuous, V::Error>
                where V: de::MapAccess<'de>
            {
                let mut lb = None;
                let mut ub = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Lb => {
                            if lb.is_some() {
                                return Err(de::Error::duplicate_field("lb"));
                            }

                            lb = Some(map.next_value()?);
                        }
                        Field::Ub => {
                            if ub.is_some() {
                                return Err(de::Error::duplicate_field("ub"));
                            }

                            ub = Some(map.next_value()?);
                        }
                    }
                }

                let lb = lb.ok_or_else(|| de::Error::missing_field("lb"))?;
                let ub = ub.ok_or_else(|| de::Error::missing_field("ub"))?;

                Ok(Continuous::new(lb, ub))
            }
        }

        deserializer.deserialize_struct("Continuous", FIELDS, ContinuousVisitor)
    }
}

impl cmp::PartialEq for Continuous {
    fn eq(&self, other: &Continuous) -> bool {
        self.lb.eq(&other.lb) && self.ub.eq(&other.ub)
    }
}

impl fmt::Debug for Continuous {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Continuous")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .finish()
    }
}


/// A finite, uniformly partitioned continous dimension.
#[derive(Clone, Copy, Serialize)]
pub struct Partitioned {
    lb: f64,
    ub: f64,
    density: usize,

    #[serde(skip_serializing)]
    range: RngRange<f64>,
}

impl Partitioned {
    pub fn new(lb: f64, ub: f64, density: usize) -> Partitioned {
        Partitioned {
            lb: lb,
            ub: ub,
            density: density,

            range: RngRange::new(lb, ub),
        }
    }

    pub fn from_continuous(d: Continuous, density: usize) -> Partitioned {
        Partitioned {
            lb: d.lb,
            ub: d.ub,
            density: density,

            range: d.range,
        }
    }

    pub fn to_partition(&self, val: f64) -> usize {
        let clipped = clip!(self.lb, val, self.ub);

        let diff = clipped - self.lb;
        let range = self.ub - self.lb;

        let i = ((self.density as f64) * diff / range).floor() as usize;

        if i == self.density { i - 1 } else { i }
    }

    pub fn centres(&self) -> Vec<f64> {
        let w = (self.ub - self.lb) / self.density as f64;
        let hw = w / 2.0;

        (0..self.density).map(|i| self.lb + w * (i as f64) - hw).collect()
    }

    pub fn partition_width(&self) -> f64 {
        (self.lb - self.ub) / self.density as f64
    }

    pub fn density(&self) -> usize {
        self.density
    }
}

impl Dimension for Partitioned {
    type Value = usize;

    fn sample(&self, rng: &mut ThreadRng) -> usize {
        self.to_partition(self.range.ind_sample(rng))
    }

    fn convert(&self, val: f64) -> Self::Value {
        self.to_partition(val)
    }

    fn span(&self) -> Span {
        Span::Finite(self.density)
    }
}

impl BoundedDimension for Partitioned {
    type ValueBound = f64;

    fn lb(&self) -> &f64 {
        &self.lb
    }

    fn ub(&self) -> &f64 {
        &self.ub
    }

    fn contains(&self, val: Self::ValueBound) -> bool {
        (val >= self.lb) && (val < self.ub)
    }
}

impl FiniteDimension for Partitioned {
    fn range(&self) -> Range<Self::Value> {
        0..(self.density + 1)
    }
}

impl<'de> Deserialize<'de> for Partitioned {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        enum Field {
            Lb,
            Ub,
            Density,
        };
        const FIELDS: &'static [&'static str] = &["lb", "ub", "density"];

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`lb`, `ub` or `density`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                        where E: de::Error
                    {
                        match value {
                            "lb" => Ok(Field::Lb),
                            "ub" => Ok(Field::Ub),
                            "density" => Ok(Field::Density),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct PartitionedVisitor;

        impl<'de> Visitor<'de> for PartitionedVisitor {
            type Value = Partitioned;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Partitioned")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Partitioned, V::Error>
                where V: de::SeqAccess<'de>
            {
                let lb = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let ub = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let density = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;

                Ok(Partitioned::new(lb, ub, density))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Partitioned, V::Error>
                where V: de::MapAccess<'de>
            {
                let mut lb = None;
                let mut ub = None;
                let mut density = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Lb => {
                            if lb.is_some() {
                                return Err(de::Error::duplicate_field("lb"));
                            }

                            lb = Some(map.next_value()?);
                        }
                        Field::Ub => {
                            if ub.is_some() {
                                return Err(de::Error::duplicate_field("ub"));
                            }

                            ub = Some(map.next_value()?);
                        }
                        Field::Density => {
                            if density.is_some() {
                                return Err(de::Error::duplicate_field("density"));
                            }

                            density = Some(map.next_value()?);
                        }
                    }
                }

                let lb = lb.ok_or_else(|| de::Error::missing_field("lb"))?;
                let ub = ub.ok_or_else(|| de::Error::missing_field("ub"))?;
                let density = density.ok_or_else(|| de::Error::missing_field("density"))?;

                Ok(Partitioned::new(lb, ub, density))
            }
        }

        deserializer.deserialize_struct("Partitioned", FIELDS, PartitionedVisitor)
    }
}

impl cmp::PartialEq for Partitioned {
    fn eq(&self, other: &Partitioned) -> bool {
        self.lb.eq(&other.lb) && self.ub.eq(&other.ub) && self.density.eq(&other.density)
    }
}

impl fmt::Debug for Partitioned {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Partitioned")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("density", &self.density)
            .finish()
    }
}


/// A finite discrete dimension.
#[derive(Clone, Copy, Serialize)]
pub struct Discrete {
    size: usize,

    #[serde(skip_serializing)]
    ub: usize,

    #[serde(skip_serializing)]
    range: RngRange<usize>,
}

impl Discrete {
    pub fn new(size: usize) -> Discrete {
        Discrete {
            ub: size - 1,
            size: size,
            range: RngRange::new(0, size),
        }
    }
}

impl Dimension for Discrete {
    type Value = usize;

    fn sample(&self, rng: &mut ThreadRng) -> usize {
        self.range.ind_sample(rng)
    }

    fn convert(&self, val: f64) -> Self::Value {
        val as usize
    }

    fn span(&self) -> Span {
        Span::Finite(self.size)
    }
}

impl BoundedDimension for Discrete {
    type ValueBound = usize;

    fn lb(&self) -> &usize {
        &0
    }

    fn ub(&self) -> &usize {
        &self.ub
    }

    fn contains(&self, val: Self::Value) -> bool {
        val < self.size
    }
}

impl FiniteDimension for Discrete {
    fn range(&self) -> Range<Self::Value> {
        0..self.size
    }
}

impl<'de> Deserialize<'de> for Discrete {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        enum Field {
            Size,
        };
        const FIELDS: &'static [&'static str] = &["size"];

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`size`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                        where E: de::Error
                    {
                        match value {
                            "size" => Ok(Field::Size),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct DiscreteVisitor;

        impl<'de> Visitor<'de> for DiscreteVisitor {
            type Value = Discrete;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Discrete")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Discrete, V::Error>
                where V: de::SeqAccess<'de>
            {
                let size = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;

                Ok(Discrete::new(size))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Discrete, V::Error>
                where V: de::MapAccess<'de>
            {
                let mut size = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }

                            size = Some(map.next_value()?);
                        }
                    }
                }

                Ok(Discrete::new(size.ok_or_else(|| de::Error::missing_field("size"))?))
            }
        }

        deserializer.deserialize_struct("Discrete", FIELDS, DiscreteVisitor)
    }
}

impl cmp::PartialEq for Discrete {
    fn eq(&self, other: &Discrete) -> bool {
        self.size.eq(&other.size)
    }
}

impl fmt::Debug for Discrete {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Discrete")
            .field("size", &self.size)
            .finish()
    }
}


impl<D: Dimension> Dimension for Box<D> {
    type Value = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Value {
        (**self).sample(rng)
    }

    fn convert(&self, val: f64) -> Self::Value {
        (**self).convert(val)
    }

    fn span(&self) -> Span {
        (**self).span()
    }
}

impl<'a, D: Dimension> Dimension for &'a D {
    type Value = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Value {
        (**self).sample(rng)
    }

    fn convert(&self, val: f64) -> Self::Value {
        (**self).convert(val)
    }

    fn span(&self) -> Span {
        (**self).span()
    }
}

// TODO: Use quickcheck here to more extenisively test calls to contains...

#[cfg(test)]
mod tests {
    use super::{Dimension, BoundedDimension, FiniteDimension};
    use super::{Null, Infinite, Continuous, Partitioned, Discrete};
    use geometry::Span;

    use rand::thread_rng;
    use serde_test::{Token, assert_tokens};

    #[test]
    fn test_null() {
        let d = Null;
        let mut rng = thread_rng();

        assert_eq!(d.sample(&mut rng), ());
        assert_eq!(d.span(), Span::Null);

        assert_tokens(&d, &[Token::UnitStruct { name: "Null" }]);
    }

    #[test]
    fn test_infinite() {
        let d = Infinite;

        assert_eq!(d.span(), Span::Infinite);

        assert_tokens(&d, &[Token::UnitStruct { name: "Infinite" }]);
    }

    #[test]
    #[should_panic]
    fn test_infinite_sample() {
        let d = Infinite;
        let mut rng = thread_rng();

        let _ = d.sample(&mut rng);
    }

    #[test]
    fn test_continuous() {
        for (lb, ub) in vec![(0.0, 5.0), (-5.0, 5.0), (-5.0, 0.0)] {
            let d = Continuous::new(lb, ub);
            let mut rng = thread_rng();

            assert_eq!(d.span(), Span::Infinite);

            assert!(!d.contains(ub));
            assert!(d.contains(lb));
            assert!(d.contains(((lb + ub) / 2.0)));

            for _ in 0..100 {
                let s = d.sample(&mut rng);
                assert!(s < ub);
                assert!(s >= lb);
                assert!(d.contains(s));
            }

            assert_tokens(&d,
                          &[Token::Struct {
                                name: "Continuous",
                                len: 2,
                            },
                            Token::Str("lb"),
                            Token::F64(lb),
                            Token::Str("ub"),
                            Token::F64(ub),
                            Token::StructEnd]);
        }
    }

    #[test]
    fn test_partitioned() {
        for (lb, ub, density) in vec![(0.0, 5.0, 5), (-5.0, 5.0, 10), (-5.0, 0.0, 5)] {
            let d = Partitioned::new(lb, ub, density);
            let mut rng = thread_rng();

            assert_eq!(d.span(), Span::Finite(density));

            assert!(!d.contains(ub));
            assert!(d.contains(lb));
            assert!(d.contains(((lb + ub) / 2.0)));

            for _ in 0..100 {
                let s = d.sample(&mut rng);
                assert!(s < density);
            }

            assert_tokens(&d,
                          &[Token::Struct {
                                name: "Partitioned",
                                len: 3,
                            },
                            Token::Str("lb"),
                            Token::F64(lb),
                            Token::Str("ub"),
                            Token::F64(ub),
                            Token::Str("density"),
                            Token::U64(density as u64),
                            Token::StructEnd]);
        }
    }

    #[test]
    fn test_discrete() {
        for size in vec![5, 10, 100] {
            let d = Discrete::new(size);
            let mut rng = thread_rng();

            assert_eq!(d.span(), Span::Finite(size));

            assert!(!d.contains(size));

            assert!(d.contains(0));
            assert!(d.contains((size - 1)));

            for _ in 0..100 {
                let s = d.sample(&mut rng);
                assert!(s < size);
            }

            assert_tokens(&d,
                          &[Token::Struct {
                                name: "Discrete",
                                len: 1,
                            },
                            Token::Str("size"),
                            Token::U64(size as u64),
                            Token::StructEnd]);
        }
    }
}
