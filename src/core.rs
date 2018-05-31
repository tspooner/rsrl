pub trait Handler<SAMPLE> {
    #[allow(unused_variables)]
    fn handle_sample(&mut self, sample: &SAMPLE) {}

    #[allow(unused_variables)]
    fn handle_terminal(&mut self, sample: &SAMPLE) {}
}

pub trait BatchHandler<SAMPLE>: Handler<SAMPLE> {
    fn handle_batch(&mut self, batch: &Vec<SAMPLE>) {
        for sample in batch.into_iter() {
            self.handle_sample(sample);
        }
    }
}
