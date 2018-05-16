pub trait Handler {
    type Sample;

    #[allow(unused_variables)]
    fn handle_sample(&mut self, sample: &Self::Sample) {}

    #[allow(unused_variables)]
    fn handle_terminal(&mut self, sample: &Self::Sample) {}
}

pub trait BatchHandler: Handler {
    fn handle_batch(&mut self, batch: &Vec<Self::Sample>) {
        for sample in batch.into_iter() {
            self.handle_sample(sample);
        }
    }
}
