# -*- coding: utf-8 -*-
module Kalman

using LinearAlgebra, Statistics
export filter, smooth, kalman

mutable struct kalman
    observation_data
    transition_matrices
    transition_covariance
    observation_matrices
    observation_covariance
    state_0
    predict_data
    predict_covariance
    filterd_data
    filterd_covariance
    smoothed_data
    smoothed_covariance
    function kalman(observation_data,
                    transition_matrices,
                    transition_covariance,
                    observation_matrices,
                    observation_covariance,
                    state_0) 
        return kalman(observation_data,
                    transition_matrices,
                    transition_covariance,
                    observation_matrices,
                    observation_covariance,
                    state_0, 
                    missing, missing, 
                    missing, missing,
                    missing, missing)
    end
end

function state_size(k::kalman)
    return size(observation_data, 1), size(transition_matrices, 2)
end

function filter(k::kalman)::kalman
    n_step = size(k.observation_data,1)
    x_dim = size(k.state_0,1)
    y = k.observation_data
    A = k.transition_matrices
    G = k.transition_covariance
    H = k.observation_matrices
    R = k.observation_covariance
    
    x̂_k1 = zeros(n_step, x_dim); x̂_k1[1,:] .= k.state_0
    x̂_k = zeros(n_step, x_dim); x̂_k[1,:] .= k.state_0
    I = diagm(ones(x_dim))
    V_k = zeros(F, (n_step, x_dim, x_dim))
    V_k1 = zeros(F, (n_step, x_dim, x_dim))
    V_k1[1,:,:] = 100.0 * G
    V_k[1,:,:] = 100.0 * G
    Q = 1.0 #tmp
    for i in 2:n_step
        #予測ステップ
        x̂_k1[i,:] = A*x̂_k[i-1,:] #.+ G*ŵ
        V_k1[i,:,:] = A*V_k[i-1,:,:]*A' + G*Q*G'
        # フィルタリングステップ
        if length(size(R)) == 3
            R_ = R[i,:,:]
        else
            R_ = R
        end
        K = V_k1[i,:,:] * H' *inv(H*V_k1[i,:,:]*H' + R_)
        e = y[i,:] - H*x̂_k1[i,:]
        x̂_k[i,:] = x̂_k1[i,:] + K*e
        V_k[i,:,:] = (I - K*H) *V_k1[i,:,:]
    end
    return kalman_filterd(k, y, x̂_k, x̂_k1, V_k, V_k1)
end


function smooth(k::kalman)
    x̂_k = k.filterd_data
    x̂_k1 = k.predict_data
    P_k = k.filterd_covariance
    P_k1 = k.predict_covariance
    A = k.kalman.transition_matrices
    n_step = size(x̂_k, 1)
    x_dim =  size(x̂_k, 2)
    x̂_N = similar(x̂_k); x̂_N[end,:] = x̂_k[end,:]
    P_N = similar(P_k); P_N[end,:,:] = P_k[end,:,:]
    for i in n_step-1:(-1):1
        F = P_k[i,:,:]*(A')*inv(P_k1[i+1,:,:])
        x̂_N[i,:] = x̂_k[i,:] + F*(x̂_N[i+1,:] - x̂_k1[i+1,:])
        P_N[i,:,:] = P_k[i,:,:] + F*(P_N[i+1,:,:] - P_k1[i+1,:,:])*F'
    end
    return  kalman_smoothed(k.observation_data, x̂_N,P_N)
end

end
