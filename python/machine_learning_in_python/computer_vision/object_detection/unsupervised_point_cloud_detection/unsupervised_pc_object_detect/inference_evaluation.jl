using PyPlot  # Import PyPlot
using Gen  # Import gen
using Random, Distributions  # Synthetic dataset generation
using Plots  # Displaying our data
using PyCall  # Python interop
using LinearAlgebra  # Chamfer distance computation
using Plotly  # Plotting grids

# We can now write the do_inference function
function inference(model, proposal, bounds, ys, amount_of_computation)
    """Function for carrying out inference with our generative model and custom proposal and importance sampling."""

    # Create a choice map for our observed point cloud
    observations = Gen.choicemap()
    N = length(ys) ÷ 3  # Compute length of ys for generative model
    for i in 1:N  # Add observed points to choice map
        observations[(:y, i)] = ys[i,:]
    end

    # Call importance_resampling to obtain a likely trace consistent with observations
    (traces, _) = Gen.importance_sampling(model, (bounds, N), observations, proposal,
                                         (ys,), amount_of_computation);
    return traces
end;

function top_five_traces(point_cloud, original_bbox, num_traces)
    # Bounds
    x_min, x_max, y_min, y_max, z_min, z_max = 0, 1, 0, 1, 0, 1
    sigma_min = 0.01
    sigma_max = 0.1
    bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max), (sigma_min, sigma_max))

    # Perform inference
    traces = inference(mv_bounding_box_model, mean_pc_proposal, bounds, point_cloud, num_traces);

    # Sort according to highest-performing traces
    zs = compute_noisy_chamfer(traces, point_cloud);
    sorted_indices = sortperm(zs)

    # Now find best bounding box parameters
    choices = [Gen.get_choices(trace) for trace in traces]
    bboxes = [(choice[:L], choice[:W], choice[:H]) for choice in choices]
    centers = [[choice[:xc], choice[:yc], choice[:zc]] for choice in choices]

    # Get best ones
    best_bboxes = [bboxes[index] for index in sorted_indices[1:5]]
    best_centers = [centers[index] for index in sorted_indices[1:5]]
    #best_zs = [zs[index] for index in sorted_indices][1:10]
    bboxes2points = [box2points(best_bboxes[i], 20, best_centers[i][1], best_centers[i][2], best_centers[i][3]) for i in 1:5]

    # Get best chamfer values
    best_zs = [zs[index] for index in sorted_indices[1:5]]
    # Now compare performance compared to original box
    volume_bboxes = [best_bboxes[i][1] * best_bboxes[i][2] * best_bboxes[i][3] for i in 1:5]

    # Now compare percent of points inside
    points_in = [evaluate_points_inside(best_bboxes[i], best_centers[i], point_cloud) for i in 1:5]

    # Get volume of original bounding box
    volume_original_bbox = original_bbox[1] * original_bbox[2] * original_bbox[3]

return best_bboxes, best_centers, best_zs, volume_bboxes, points_in, volume_original_bbox
end

# Evaluation on synthetic data
function evaluate_synthetic_data(synthetic_bounding_boxes, synthetic_point_clouds, num_traces)
    """Evaluate inference results on synthetically-generated data."""
    i = 1

    # Iterate through synthetically-generated point cloud datasets.
    for (bbox, points) in zip(synthetic_bounding_boxes, synthetic_point_clouds)
        best_bboxes, best_centers, best_zs, volume_bboxes, points_in, volume_original_bbox = top_five_traces(points, bbox, num_traces)
        println("ITERATION $i")
        println("Best bboxes: $best_bboxes")
        println("Best centers: $best_centers")
        println("Best zs: $best_zs")
        println("Best volume: $volume_bboxes")
        println("Points in: $points_in")
        println("Volume original bbox: $volume_original_bbox")
        println("____________________________________________")
        i += 1
    end
end

# Evaluation on A2D2 dataset
function evaluate_real_data(bounding_boxes, point_clouds)
    """Function for performing inference on our A2D2 dataset."""
    for (bbox, points) in zip(bounding_boxes, point_clouds)
        best_bboxes, best_centers, best_zs, volume_bboxes, points_in, volume_original_bbox = top_five_traces(points, bbox)
    end
end