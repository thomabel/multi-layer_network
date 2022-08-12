//use ndarray::prelude::*;
use plotters::prelude::*;
use crate::utility::epoch::Results;

pub fn accuracy_graph_png(result: &Results, title: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("./visuals/{}.png", title);
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let epochs = result.accuracy.len();
    let x_spec = 0.0..epochs as f32;
    let y_spec = 0.0..1.0f32;

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_spec, y_spec)?;

    chart.configure_mesh().draw()?;

    let iter = result.accuracy.iter().enumerate()
        .map(|x| (x.0 as f32, *x.1));
    chart.draw_series(LineSeries::new(iter,RED.filled()).point_size(2))?;
    root.present()?;

    Ok(())
}
