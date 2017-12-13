extern crate rsrl;
extern crate serde_json;

use std::fs::File;
use rsrl::{run, logging, Parameter, SerialExperiment, Evaluation};
use rsrl::agents::control::td::ExpectedSARSA;
use rsrl::domains::{Domain, HIVTreatment};
use rsrl::fa::Linear;
use rsrl::fa::projection::Fourier;
use rsrl::geometry::Space;
use rsrl::policies::EpsilonGreedy;


fn main() {
    let logger = logging::root(logging::stdout());

    let domain = HIVTreatment::default();
    let mut agent = {
        let aspace = domain.action_space();
        let n_actions: usize = aspace.span().into();

        let pr = Fourier::from_space(3, domain.state_space());
        let q_func = Linear::new(pr, n_actions);
        let policy = EpsilonGreedy::new(Parameter::exponential(0.99, 0.05, 0.99));

        ExpectedSARSA::new(q_func, policy, 0.2, 0.99)
    };

    // Training:
    let _training_result = {
        let e = SerialExperiment::new(&mut agent, Box::new(HIVTreatment::default), 200);

        run(e, 200, Some(logger))
    };

    serde_json::to_writer_pretty(File::create("/tmp/q.json").unwrap(), &agent.q_func);

    // Testing:
    let testing_result =
        Evaluation::new(&mut agent, Box::new(HIVTreatment::default)).next().unwrap();


    println!("Solution \u{21D2} {} steps | reward {}",
             testing_result.steps,
             testing_result.reward);
}
