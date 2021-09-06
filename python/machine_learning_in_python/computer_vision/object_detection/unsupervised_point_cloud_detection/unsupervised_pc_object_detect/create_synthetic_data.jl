using PyPlot  # Import PyPlot
using Gen  # Import gen
using Random, Distributions  # Synthetic dataset generation
using Plots  # Displaying our data
using PyCall  # Python interop
using LinearAlgebra  # Chamfer distance computation
using Plotly  # Plotting grids

# First, set random seed
rng = MersenneTwister(8080);

function pc_model_uniform(N::Int)
    """ Generative model for our point cloud synthetic dataset. Samples random 3-dimensional points in
    the domain [0, 1].
    """
    return rand!(rng, zeros(N, 3))
end

function pc_model_bbox_random(N, L, W, H, sigma, xc=0.0, yc=0.0, zc=0.0)
    """ Generative model for our point cloud synthetic dataset. Samples random 3-dimensional points using Gaussian
    distributions centered on the coordinates of a specified bounding box.
    """
    # Compute points per segment
    N_seg = convert(Int32, N/12)

    # Get origin
    x0 = xc - L/2
    y0 = yc - W/2
    z0 = zc - H/2

    # First group of segments
    s1 = [[x0+(i*L)/N_seg, y0, z0]+rand(Normal(0, sigma), 3) for i in 1:N_seg]
    s2 = [[x0+(i*L)/N_seg, y0+W, z0]+rand(Normal(0, sigma), 3) for i in 1:N_seg]
    s3 = [[x0+(i*L)/N_seg, y0, z0+H]+rand(Normal(0, sigma), 3) for i in 1:N_seg]
    s4 = [[x0+(i*L)/N_seg, y0+W, z0+H]+rand(Normal(0, sigma), 3) for i in 1:N_seg]

    # Second group of segments
    s5 = [[x0, y0+(j*W)/N_seg, z0]+rand(Normal(0, sigma), 3) for j in 1:N_seg]
    s6 = [[x0+L, y0+(j*W)/N_seg, z0]+rand(Normal(0, sigma), 3) for j in 1:N_seg]
    s7 = [[x0, y0+(j*W)/N_seg, z0+H]+rand(Normal(0, sigma), 3) for j in 1:N_seg]
    s8 = [[x0+L, y0+(j*W)/N_seg, z0+H]+rand(Normal(0, sigma), 3) for j in 1:N_seg]

    # Third group of segments
    s9 = [[x0, y0, z0+(k*H)/N_seg]+rand(Normal(0, sigma), 3) for k in 1:N_seg]
    s10 = [[x0+L, y0, z0+(k*H)/N_seg]+rand(Normal(0, sigma), 3) for k in 1:N_seg]
    s11 = [[x0, y0+W, z0+(k*H)/N_seg]+rand(Normal(0, sigma), 3) for k in 1:N_seg]
    s12 = [[x0+L, y0+W, z0+(k*H)/N_seg]+rand(Normal(0, sigma), 3) for k in 1:N_seg]

    segments = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
    A = zeros(12*N_seg+1, 3)
    A[1,:] = [0 0 0]
    index = 2
    S_tot = zeros(3)
    for s in segments
        for si in s
            A[index,:] = si
            S_tot += si
            index += 1
        end
    end
    mean = S_tot*(1/N)
    return A
end

function box2points(dims, points_per_seg, xc=0.0, yc=0.0, zc=0.0)
    """Function for transforming a given bounding box (determined by length (L), width (W), and height (H),
    as well as a center point given by (xc, yc, zc).  The total number of points in this point cloud is equal
    to (12*points_per_seg) + 1."""

    # Get dimensions of box
    (L, W, H) = dims

    # Get origin
    x0 = xc - L/2
    y0 = yc - W/2
    z0 = zc - H/2

    # First group of segments
    s1 = [[x0+((i*L)/points_per_seg), y0, z0] for i in 1:points_per_seg]
    s2 = [[x0+((i*L)/points_per_seg), y0+W, z0] for i in 1:points_per_seg]
    s3 = [[x0+((i*L)/points_per_seg), y0, z0+H] for i in 1:points_per_seg]
    s4 = [[x0+((i*L)/points_per_seg), y0+W, z0+H] for i in 1:points_per_seg]

    # Second group of segments
    s5 = [[x0, y0+((j*W)/points_per_seg), z0] for j in 1:points_per_seg]
    s6 = [[x0+L, y0+((j*W)/points_per_seg), z0] for j in 1:points_per_seg]
    s7 = [[x0, y0+((j*W)/points_per_seg), z0+H] for j in 1:points_per_seg]
    s8 = [[x0+L, y0+((j*W)/points_per_seg), z0+H] for j in 1:points_per_seg]

    # Third group of segments
    s9 = [[x0, y0, z0+((k*H)/points_per_seg)] for k in 1:points_per_seg]
    s10 = [[x0+L, y0, z0+((k*H)/points_per_seg)] for k in 1:points_per_seg]
    s11 = [[x0, y0+W, z0+((k*H)/points_per_seg)] for k in 1:points_per_seg]
    s12 = [[x0+L, y0+W, z0+((k*H)/points_per_seg)] for k in 1:points_per_seg]

    # Concatenate all segments
    segments = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]

    # Initialize output data structure
    A = zeros(12*points_per_seg+1, 3)
    A[1,:] = [0 0 0]
    index = 2
    for s in segments  # Add segments
        for si in s  # Add points in segments
            A[index,:] = si
            index += 1
        end
    end
    return A
    end

