//use ndarray::prelude::*;
use plotters::prelude::*;
use crate::utility::epoch;

pub fn accuracy_graph_png(name: &str, result: &(epoch::Results, epoch::Results)) -> Result<(), Box<dyn std::error::Error>> {
    // Set up the graph file.
    let path = format!("./visuals/{}.png", name);
    let size = (1280, 720);
    let root = BitMapBackend::new(&path, size).into_drawing_area();
    root.fill(&WHITE)?;

    // Get info for graph dimensions.
    let epochs = result.0.accuracy.len();
    let x_spec = 0.0..epochs as f32;
    let y_spec = 0.0..1.0f32;

    // Create the graph.
    let mut chart = ChartBuilder::on(&root)
        .caption(&name, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_spec, y_spec)?;

    chart.configure_mesh().draw()?;

    // Plot the train set line.
    let iter
        = result.0.accuracy.iter().enumerate()
        .map(|x| (x.0 as f32, *x.1));
    chart.draw_series(LineSeries::new(iter, BLUE.filled()).point_size(2))?
        .label("Train")
        .legend(|(x, y)| 
        PathElement::new(vec![(x, y), (x + 20, y)], 
        BLUE));

    // Plot the test set line.
    let iter 
        = result.1.accuracy.iter().enumerate()
        .map(|x| (x.0 as f32, *x.1));
    chart.draw_series(LineSeries::new(iter, RED.filled()).point_size(2))?
    .label("Test")
    .legend(|(x, y)| 
        PathElement::new(vec![(x, y), (x + 20, y)], 
        RED));

    // Create the line key.
    chart
    .configure_series_labels()
    .background_style(&WHITE.mix(0.8))
    .border_style(&BLACK)
    .draw()?;

    root.present()?;
    Ok(())
}
