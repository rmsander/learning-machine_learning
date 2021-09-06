using PyPlot  # Import PyPlot
using Gen  # Import gen
using Random, Distributions  # Synthetic dataset generation
using Plots  # Displaying our data
using PyCall  # Python interop
using LinearAlgebra  # Chamfer distance computation
using Plotly  # Plotting grids
using NPZ  # Loading A2D2 data
using JSON  # Loading bounding box label data from A2D2 dataset

function get_files(path)
    """Load A2D2 files from a given directory of the A2D2 date sub-directories."""

    # Get data for lidar
    lidar_path = string(path, "lidar/cam_front_center/")
    lidar_files = readdir(lidar_path)
    lidar_paths = [string(lidar_path, lidar_file) for lidar_file in lidar_files]

    # Get data for lidar
    bbox_path = string(path, "label3D/cam_front_center/")
    bbox_files = readdir(bbox_path)
    bbox_paths = [string(bbox_path, bbox_file) for bbox_file in bbox_files]

    return lidar_paths, bbox_paths
end

function load_data(lidar_path)
    """Load data using a NumPy file reader."""

    # Initialize output point clouds
    PC = npzread(lidar_path)["points"]
    return PC
end

using JSON
function load_bbox(bbox_path)
    """Load bounding box from a given bounding box path."""
    pc = nothing
    open(bbox_path, "r") do f
        dicttxt = JSON.parse(f)  # file information to string
        boxes = keys(dicttxt)
        box_points = [dicttxt[box]["3d_points"] for box in boxes]
        box_classes = [dicttxt[box]["class"] for box in boxes]
        for (j, box_class) in enumerate(box_classes)
            if box_class == "Car"
                pc = [convert(Array{Float64,1}, box_points[j][i]) for i in 1:length(box_points[j])]
                break
            end

        end
    end
    return pc
end

function normalize_and_crop(bbox_points)
    """Normalize all points to fall in the interval [0, 1]^3, according to the locations of the bounding box
    point cloud values."""

    # Compute the mean of the observations
    x = getindex.(bbox_points, 1)
    y = getindex.(bbox_points, 2)
    z = getindex.(bbox_points, 3)

    min_x, max_x = minimum(x), maximum(x)
    delta_x = max_x-min_x
    min_y, max_y = minimum(y), maximum(y)
    delta_y = max_y-min_y
    min_z, max_z = minimum(z), maximum(z)
    delta_z = max_z-min_z


    x_norm = [(xi-min_x)/(2*delta_x) for xi in x]
    y_norm = [(yi-min_y)/(2*delta_y) for yi in y]
    z_norm = [(zi-min_z)/(2*delta_z) for zi in z]

    println(x_norm)
    norm_data = zeros(length(x), 3)

    i = 1
    for (x,y,z) in zip(x_norm, y_norm, z_norm)
        norm_data[i,:] = [x,y,z]
        i += 1
    end

    return norm_data
end
