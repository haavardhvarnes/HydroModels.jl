"""
Time-handling utilities — stages, load blocks, hour conversions.
"""

# Default Nordic convention: weekly stages
const HOURS_PER_WEEK = 168.0
const WEEKS_PER_YEAR = 52

"""
    stage_hours(num_load_blocks::Int, hours_per_stage=168.0)

Default equal-duration load blocks summing to a weekly stage.
"""
function stage_hours(num_load_blocks::Int, hours_per_stage::Real = HOURS_PER_WEEK)
    return fill(hours_per_stage / num_load_blocks, num_load_blocks)
end

# TODO: scheduling-aware date arithmetic, daylight-saving handling,
# leap year corrections, etc.
