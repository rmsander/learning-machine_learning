using PyPlot  # Import PyPlot
using Gen  # Import gen
using Random, Distributions  # Synthetic dataset generation
using Plots  # Displaying our data
using PyCall  # Python interop
using LinearAlgebra  # Chamfer distance computation
using Plotly  # Plotting grids

@gen function mv_bounding_box_model(bounds, N, zeta=0.01)
    """Generative model for our bounding box using Gen.  This generative function samples a center point of the
    point cloud (xc, yc, zc), as well as dimensions length (L), width (W), and height (H).  It also samples a
    standard deviation sigma.  With all of these variables, it then samples points from the proposed center point
    according to the multivariate normal distribution N([xc, yc, zc], sigma*I3)."""

    # Get bounds from arguments
    ((x_min, x_max), (y_min, y_max), (z_min, z_max), (sigma_min, sigma_max)) = bounds

    # Sample center coordinate of the bounding box
    xc = @trace(uniform(x_min, x_max), :xc)
    yc = @trace(uniform(y_min, y_max), :yc)
    zc = @trace(uniform(z_min, z_max), :zc)

    # Sample L, W, H
    W = @trace(uniform(0, 1), :W)
    L = @trace(uniform(0, 1), :L)
    H = @trace(uniform(0, 1), :H)

    # Sample standard deviation for the center of the point cloud
    sigma = @trace(uniform(sigma_min, sigma_max), :sigma)

    mean = [xc, yc, zc]
    cov = Matrix(sigma*I, 3, 3)

    # Now sample N points using a multivariate normal distribution
    sampled_points = zeros(N, 3)
    for i in 1:N
        sampled_points[i, :] = @trace(mvnormal(mean, Matrix(sigma*I, 3, 3)), (:y, i))
    end
    # We also use the returned y values
end

@gen function mean_pc_proposal(observations, zeta=0.01)
    """Gen proposal function for more efficiently proposing point clouds.  Samples the mean of the proposed point
    cloud at three-dimensional locations that are noisy means of the true location of the observed point cloud.
    Effectively, this ensures that the proposed bounding box is centered around the observed point cloud.  This
    is used as a custom proposal distribution in our importance sampling method."""

    # Compute the mean of the observations
    mu = sum(observations, dims=1) / (length(observations) ÷ 3)

    # Sample the center point of the point cloud
    xc = @trace(normal(mu[1], zeta), :xc)
    yc = @trace(normal(mu[2], zeta), :yc)
    zc = @trace(normal(mu[3], zeta), :zc)

    return nothing
end
