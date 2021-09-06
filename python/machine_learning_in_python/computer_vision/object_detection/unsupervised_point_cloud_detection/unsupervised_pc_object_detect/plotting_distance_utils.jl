using PyPlot  # Import PyPlot
using Gen  # Import gen
using Random, Distributions  # Synthetic dataset generation
using Plots  # Displaying our data
using PyCall  # Python interop
using LinearAlgebra  # Chamfer distance computation
using Plotly  # Plotting grids

################################################################################
# PLOTTING
################################################################################

function plot_data(points1, points2, title="")
    """Function for plotting point cloud data, usually comparing a point-transformed bounding box to an
    observed point cloud.  For visualizing a single point cloud, simply pass the same set of points in twice."""

    Plots.scatter(points2[:,1], points2[:,2], points2[:,3], label="Bounding Box", markersize=1, title=title)
    Plots.scatter!(points1[:,1], points1[:,2],points1[:,3], label="Point Cloud", markersize=3)
end

function plot_grid(traces, point_cloud, points_per_seg)

    # Iterate through traces
    for (i,trace) in enumerate(traces)

        # Get choices from traces
        choices = Gen.get_choices(trace)

        # Get parameters
        L1 = choices[:L]
        W1 = choices[:W]
        H1 = choices[:H]
        # Convert to points
        box_points = box2points((L1, W1, H1), points_per_seg)

        # Create scatterplots
        plt = Plots.scatter(box_points[:,1],box_points[:,2],box_points[:,3], label="Bounding Box", markersize=1, title= "z =$choices[:z]")
        Plots.scatter!(point_cloud[:,1],point_cloud[:,2],point_cloud[:,3], label="Point Cloud", markersize=2)
        Plots.display(plt)
    end
end

function plot_hist_and_time_series(zs)
    """Function for plotting the histogram and time series of data over Chamfer distance values for a set of traces."""
    p1 = Plots.histogram(zs, bins=10, title="Z across traces")
    step = 10
    p2 = Plots.plot([i for i in 1:step:length(zs)], [zs[i] for i in 1:step:length(zs)], title="Z vs. number of resampled traces")
    Plots.plot(p1, p2, layout = (1, 2), legend = false)
end

################################################################################
# METRICS
################################################################################

# Implementation source: http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf
function chamfer_distance(points_A, points_B)
    """Function for computing the Chamfer distance between two point clouds ('points_A' and 'points_B').  Note: this metric is
    not normalized by the total number of points in the point cloud."""

    # Used to keep track of total chamfer distance
    tot = 0

    # Compute error for point cloud A
    for pA in points_A
        X = [pA-pB for pB in points_B]
        tot += minimum([abs2(LinearAlgebra.norm(X[i,:], 2)) for i in 1:convert(Int32, length(X)/3)])
    end

function compute_noisy_chamfer(traces, observed_points)
    """Compute Chamfer distance between proposed traces and and observed points."""

    # Get sampled choice values from traces
    choices = [Gen.get_choices(trace) for trace in traces]
    bboxes = [(choice[:L], choice[:W], choice[:H]) for choice in choices]

    # Initialize output Chamfer distance values
    zs = zeros(length(traces))
    N_observed_points = length(observed_points) ÷ 3

    # Iterate through bounding boxes from traces
    for (i, bbox) in enumerate(bboxes)
        if i % 100 == 0
            println("Iterated through $i traces")
        end
        bbox_points = box2points(bbox, N_observed_points ÷ 12)
        zs[i] = chamfer_distance(bbox_points, observed_points)
    end
    return zs
end

    # Compute error for point cloud B
    for pB in points_B
        X = [pB-pA for pA in points_A]
        tot += minimum([abs2(LinearAlgebra.norm(X[i,:], 2)) for i in 1:convert(Int32, length(X)/3)])
    end
    return tot
end

