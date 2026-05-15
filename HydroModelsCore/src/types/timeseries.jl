"""
Time-series containers used by SHOP-style operational types.

These wrap an `OrderedDict{DateTime, T}` so that ordering by time is
preserved across parsing, alignment, and serialization. The forecasting
layer (`HydroModelsForecast`) and the data layer (`HydroModelsData`)
both produce instances of these types.
"""

"""
    InflowSeries{T}

Reservoir inflow time series, indexed by `DateTime` keys in insertion
order. Units are problem-specific (typically m³/s or Mm³/period).
"""
struct InflowSeries{T}
    by_time::OrderedDict{DateTime, T}
end

InflowSeries{T}() where {T} = InflowSeries{T}(OrderedDict{DateTime, T}())

Base.length(s::InflowSeries) = length(s.by_time)
Base.isempty(s::InflowSeries) = isempty(s.by_time)

"""
    MarketSeries{T}

Market-price time series (e.g. EUR/MWh), indexed by `DateTime` keys in
insertion order.
"""
struct MarketSeries{T}
    by_time::OrderedDict{DateTime, T}
end

MarketSeries{T}() where {T} = MarketSeries{T}(OrderedDict{DateTime, T}())

Base.length(s::MarketSeries) = length(s.by_time)
Base.isempty(s::MarketSeries) = isempty(s.by_time)
