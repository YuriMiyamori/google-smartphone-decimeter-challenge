# -*- coding: utf-8 -*-
module util_fc
using DataFrames, DataFramesMeta, CategoricalArrays, JLD2, Dates, Combinatorics, Statistics, Format, CSV, Glob, Formatting, GeographicLib, LinearAlgebra, LsqFit
using JLD2

F = Float64
S = Symbol

export  lnglat2XY, XY2lnglat, XY2acce, XY22acce, dist, calc_score, calc_err,build_score_df, shift_n, get_gdf_by_key, mix_pos

function get_gdf_by_key(gd::GroupedDataFrame, key::Tuple{Any})::Union{DataFrame, Missing}
    if haskey(gd.keymap, key)
        num_gdf = gd.keymap[key]
        return gd[num_gdf]
    else
        return missing
    end
end

function lnglat2XY(df::DataFrame, lng::S, lat::S, X::S, Y::S)
    ## 中心点は測定されたlng,latとする，ground truth は中心に使用しない
    df[!, X] = similar(df[!,lng],F)
    df[!, Y] = similar(df[!,lat],F)
    lngDeg_centre = mean(df[!,:lngDeg]) #! あくまで中心点はlngDegの中心，groundtruthでも同じ
    latDeg_centre = mean(df[!,:latDeg]) #! あくまで中心点はlngDegの中心，groundtruthでも同じ
    df[!, :lngDeg_centre] .= lngDeg_centre
    df[!, :latDeg_centre] .= latDeg_centre
    gd = groupby(df, :phone)
    for gdf in gd
        for (i, (lat_, lng_)) in enumerate(zip(gdf[!,lat], gdf[!,lng]))
            res = inverse(lngDeg_centre, latDeg_centre, lng_, lat_)
            dist, θ = res[:dist], deg2rad(res[:azi])
            gdf[i, X], gdf[i, Y] = dist*sin(θ), dist*cos(θ) 
        end
    end
end

function XY2lnglat(df::DataFrame, X::S, Y::S)
    # lng0, lat0からの距離x,yをもとにlng, latを返す
    lng = S("lngDeg_$(X)")
    lat = S("latDeg_$(Y)")
    df[!, lng] = similar(df[!,X])
    df[!, lat] = similar(df[!,Y])
    gd = groupby(df, :phone)
    for gdf in gd
        x = gdf[:,X] |> Array
        y = gdf[:,Y] |> Array
        θ = atand.(x, y)
        d = sqrt.(x.^2 .+ y.^2)
        t= forward.(Array(gdf[:,:lngDeg_centre]), Array(gdf[:,:latDeg_centre]), θ, d)
        gdf[:, lng] = get.(t, :lon, NaN)
        gdf[:, lat] = get.(t, :lat, NaN)
    end
end

function XY2Vel(X::Vector{F}, Y::Vector{F})
    ΔT = 1.0
    n = size(X,1)
    Vx = similar(X, F)
    Vy = similar(Y, F)
    V  = similar(Vx, F)
    for i in 1:n-1
        Vx[i] = (-X[i] + X[i+1]) / ΔT
        Vy[i] = (-Y[i] + Y[i+1]) / ΔT
        V[i] = sqrt(Vx[i]^2 + Vy[i]^2)
    end
    Vx[n] = Vx[n-1]
    Vy[n] = Vy[n-1]
    V[n] = V[n-1]

    return Vx, Vy, V

end

function XY2Vel(df::DataFrame, X::S, Y::S)
    Vx = S("Vx_$(X)")
    Vy = S("Vy_$(Y)")
    df[:, Vx] = similar(df[:, X], F)
    df[:, Vy] = similar(df[:, Y], F)
    gd = groupby(df, :phone)
    for gdf in gd
        Vx_, Vy_, V_ = XY2Vel(gdf[:,X] |> Array, gdf[:, Y] |> Array)
        # gdf[:, acce] = ac
        gdf[:, Vx] = Vx_
        gdf[:, Vy] = Vy_
        # acx, acy, ac = acce(gdf.X_klmn |> Array, gdf.Y_klmn |> Array)
    end
end
function XY2acce(X::Vector{F}, Y::Vector{F})
    ΔT = 1.0
    n = size(X,1)
    acx = similar(X, F)
    acy = similar(Y, F)
    ac = similar(acx, F)
    for i in 2:n-1
        acx[i] = (X[i-1] - 2.0*X[i] + X[i+1]) / ΔT^2
        acy[i] = (Y[i-1] - 2.0*Y[i] + Y[i+1]) / ΔT^2
        ac[i] = sqrt(acx[i]^2 + acy[i]^2)
    end
    ac[1] = ac[2]
    ac[n] = ac[n-1]

    return acx, acy, ac

end

function XY2acce(df::DataFrame, X::S, Y::S)
    Ax= S("Ax_$(X)")
    Ay= S("Ay_$(Y)")
    # df[:, A] = similar(df[:, X], F)
    df[:, Ax] = similar(df[:, X], F)
    df[:, Ay] = similar(df[:, X], F)
    gd = groupby(df, :phone)
    for gdf in gd
        acx, acy, ac = XY2acce(gdf[:,X] |> Array, gdf[:, Y] |> Array)
        # gdf[:, acce] = ac
        gdf[:, Ax] = acx
        gdf[:, Ay] = acy
        # acx, acy, ac = acce(gdf.X_klmn |> Array, gdf.Y_klmn |> Array)
    end
end

function XY22acce(X::Vector{F}, Y::Vector{F})
    ΔT = 1.0
    n = size(X,1)
    acx = similar(X, F)
    acy = similar(Y, F)
    ac = similar(acx, F)
    for i in 3:n-2
        acx[i] = (X[i-2] - 2.0*X[i] + X[i+2]) / 2ΔT^2
        acy[i] = (Y[i-2] - 2.0*Y[i] + Y[i+2]) / 2ΔT^2
        ac[i] = sqrt(acx[i]^2 + acy[i]^2)
    end
    ac[2] = ac[3]
    ac[1] = ac[2]
    ac[n-1] = ac[n-2]
    ac[n] = ac[n-1]

    return acx, acy, ac

end

function XY22acce(df::DataFrame, X::S, Y::S, acce::S)
    df[:, acce] = similar(df[:, X], F)
    gd = groupby(df, :phone)
    for gdf in gd
        acx, acy, ac = XY22acce(gdf[:,X] |> Array, gdf[:, Y] |> Array)
        gdf[:, acce] = ac
        # acx, acy, ac = acce(gdf.X_klmn |> Array, gdf.Y_klmn |> Array)
    end
end

function XYdist(x0::F, y0::F, x1::F, y1::F)::F
    return (x0 - x1)^2 +  (y0 - y1)^2 |> sqrt
end
XYdist(x1::F, y1::F ) =  dist(0.0, 0.0, x1, y1)

function lnglat_dist(lng::F, lat::F, lng_gt::F,lat_gt::F)::F
    t = inverse(lng, lat, lng_gt, lat_gt)
    dist = get(t, :dist, NaN)
    return dist
end

function shift_n(df::DataFrame, col::S, n::Int)
    out = similar(df[!, col], F) 
    out_col =  S("$(col)_m$(n)")
    out[1:end-n] .= df[1+n:end, col]
    out[end-n+1:end] .= df[end-n, col]
    df[!, out_col] = out
end

function lnglat_dist(df::DataFrame, lng::S, lat::S, out=missing::Union{Missing, Symbol})
    dist = lnglat_dist.(df[!, lng], df[!, lat], df[!, :lngDeg_gt], df[!, :latDeg_gt])
    if ismissing(out)
        return dist
    else
        df[!, out] = dist
    end
end

function plus(x::F, y::Union{F, Missing})::F
    if ismissing(y)
        return x
    else
        return x + y
    end
end

function linear_interpolation(times::Vector{DateTime}, data_times::Vector{DateTime}, datas::Vector{F})::Vector{Union{F, Missing}}
    #dataはソートされていることが前提
    n = length(times)
    m = length(data_times)
    res = Vector{Union{F, Missing}}(missing, n)
    p = sortperm(data_times)
    data_times_sort = data_times[p]
    datas_sort = datas[p]
    
    for i in 1:n
        t= times[i]
        idx0 = searchsortedlast(data_times_sort, t)
        idx1 = searchsortedfirst(data_times_sort, t)
        m < idx1 && continue
        # idx0 == 0 && println("zero, $time_arr")
        idx0 == 0 && continue
        if idx0 == idx1 
            res[i] = datas_sort[idx0]
            continue
        end
        time_delta = (data_times_sort[idx1] - data_times_sort[idx0]).value |> F
        time_delta0 = (t - data_times_sort[idx0]).value |> F
        time_delta1 = time_delta - time_delta0
        res[i] = (datas_sort[idx0] * time_delta1 + datas_sort[idx1] * time_delta0) / time_delta
        if isnan(res[i])
            res[i] = datas_sort[idx0]
        end
    end
    return res
end

function mix_pos(df::DataFrame, lng_x::S, lat_y::S, weight::S)
    df[!, S("$(lng_x)_mix")] .= 0.0
    df[!, S("$(lat_y)_mix")] .= 0.0
    df[!, :total_weight] .= 0.0
    gd1 = groupby(df, :collectionName)
    for gdf1 in gd1
        gd2 = groupby(gdf1, :phone)
        if length(gd2) == 1
            gd2[1][:, S("$(lng_x)_mix")] = gd2[1][:, lng_x]
            gd2[1][:, S("$(lat_y)_mix")] = gd2[1][:, lat_y]
            continue
        end
        for (i,gdf2) in enumerate(gd2)
            base_lng_x = Array(gdf2[:, lng_x]) .* Array(gdf2[:, weight])
            base_lat_y = Array(gdf2[:, lat_y]) .* Array(gdf2[:, weight])
            total_weight = Array(gdf2[:, weight])
            for (j, gdf3) in enumerate(gd2)
                j == i && continue
                lng_x_ = linear_interpolation(Array(gdf2[:, :time_UTC]), Array(gdf3[:, :time_UTC]), Array(gdf3[:, lng_x]))
                lat_y_ = linear_interpolation(Array(gdf2[:, :time_UTC]), Array(gdf3[:, :time_UTC]), Array(gdf3[:, lat_y]))
                w_ = linear_interpolation(Array(gdf2[:, :time_UTC]), Array(gdf3[:, :time_UTC]), Array(gdf3[:, weight]))
                base_lng_x .= plus.(base_lng_x, lng_x_ .* w_)
                base_lat_y .= plus.(base_lat_y, lat_y_ .* w_)
                total_weight .= plus.(total_weight, w_)
            end
            gdf2[:, S("$(lng_x)_mix")] = base_lng_x ./ total_weight
            gdf2[:, S("$(lat_y)_mix")] = base_lat_y ./ total_weight
            gdf2[:, :total_weight] = total_weight
        end
    end
end

function calc_err(pre_lng::F, pre_lat::F, gt_lng::F, gt_lat::F)::Vector{F}
    n = size(pre_lng, 1)
    x = 1:n #|> collect
    t = inverse(pre_lng, pre_lat, gt_lng, gt_lat)
    err = get(t, :dist, NaN)
    θ = deg2rad(get(t, :dist, NaN))
    x_err, y_err = err * sin(θ), err * cos(θ)
    return [err, x_err, y_err]
end

function calc_score(pre_lng::Vector{F}, pre_lat::Vector{F}, gt_lng::Vector{F}, gt_lat::Vector{F})::Tuple{F,F,F}
    n = size(pre_lng, 1)
    x = 1:n #|> collect
    t = calc_err.(pre_lng, pre_lat, gt_lng, gt_lat)
    err = get.(t,1, NaN)
    err_s = sort(err)
    p_50 = n - div(n, 2) + mod(n, 2)
    p_95 = n - div(n, 20)
    d_50 = err_s[p_50]
    d_95 = err_s[p_95]
    score = 0.5*(d_50 + d_95)
    return score, d_50, d_95
end

function calc_score(df::DataFrame, sub_lng::S, sub_lat::S)
    score = F[]
    d_95= F[]
    d_50= F[]
    gd = groupby(df, :phone)
    # score_df = DataFrame(phone=String[], d_50=F[], d_95=F[], score=F[])
    for gdf in gd
        score_, d_50_, d_95_ = calc_score(gdf[:, sub_lng], gdf[:, sub_lat], gdf[:, :lngDeg_gt], gdf[:, :latDeg_gt])
        push!(score,score_)
        push!(d_95,d_95_)
        push!(d_50,d_50_)
    end
    return mean(score), mean(d_95), mean(d_50)
end

function build_score_df(df::DataFrame, sub_lng::S, sub_lat::S)::DataFrame
    df = @select(df, :collectionName,  :phone, cols(sub_lng), cols(sub_lat), :lngDeg_gt, :latDeg_gt)
    calc_score(df, sub_lng, sub_lat)
    score_df = @by(df, :phone, score=mean(:phone_score), d_95=mean(:phone_d_95), d_50=mean(:phone_d_50))
    score_df = @orderby(score_df, :score .*(-1))
    return score_df
end

function write_sub(df::DataFrame, lng::S, lat::S, fname::String)
    @load "../DataFrames/sample_sub_df.jdl2" sample_sub_df
    sub_df = leftjoin(sample_sub_df, df[!, [:phone, :millisSinceGpsEpoch, lng, lat]], on = [:phone, :millisSinceGpsEpoch])
    @select!(sub_df, :phone, :millisSinceGpsEpoch, cols(lng), cols(lat))
    rename!(sub_df,lng => :lngDeg)
    rename!(sub_df,lat => :latDeg)
    open("../submit/$(fname)", "w") do io
        CSV.write(io, sub_df)
    end
end

end 
