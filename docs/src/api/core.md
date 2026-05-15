# `HydroModelsCore` — API reference

The Core layer hosts physical types, problem formulations,
uncertainty containers, the topology builder, the backend
abstraction, and generic utilities. No solver code.

## Physical types — ProdRisk style

```@docs
HydroModule
TurbineUnit
ReservoirType
RegulationReservoir
BufferReservoir
ModuleConstraints
PenaltyValues
WaterTopology
```

## Physical types — SHOP style

```@docs
Plant
Reservoir
Generator
Pump
Tunnel
Junction
```

## Time-series and market containers

```@docs
InflowSeries
MarketSeries
ElectricityMarket
LoadBlock
```

## Reserve products

```@docs
ReserveSpec
ReserveGroup
```

## Uncertainty model containers

```@docs
UncertaintyModel
ScenarioTree
ScenarioNode
StagewiseIndependent
MarkovianUncertainty
ProdRiskUncertainty
```

## Problem types

```@docs
AbstractOptProblem
LongTermHydroProblem
ShortTermHydroProblem
ShopShortTermProblem
StochasticShortTermProblem
FundamentalMarketProblem
EndValueDescription
```

## Risk measures

```@docs
AbstractRiskMeasure
Expectation
CVaR
NestedCVaR
```

## Solutions and cut representations

```@docs
Solution
WaterValueCuts
BendersCut
ReservoirWaterValue
CutGroup
```

## Topology

```@docs
build_topology
```

## Backend abstraction

```@docs
ComputeBackend
CPUBackend
GPUBackend
default_backend
set_default_backend!
allocate
zeros_like
ones_like
is_gpu
ka_backend
```

## Utilities

```@docs
PiecewiseLinear
PWL
evaluate
slopes
```
