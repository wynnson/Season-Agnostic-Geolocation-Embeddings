class Meta:
    PATH_TO_METADATA        = "meta.csv"

    INDEX                   = "Index"
    PATH                    = "Path"
    SEASON                  = "Season"
    SUMMER                  = "Summer"
    SNOW                    = "Snow"
    ALLOWED_SEASONS         = {SUMMER, SNOW}

    GRID                    = "Grid"
    LATITUDE                = "Latitude"
    LONGITUDE               = "Longitude"

    CLOUDS                  = "Clouds"
    SNOW_COVER              = "Snow"


class Pairs:
    LOC_ID                  = "loc_id"
    PATH_SUMMER             = "Path_summer"
    PATH_SNOW               = "Path_snow"
    LONGITUDE_SUMMER        = "Longitude_summer"
    LATITUDE_SUMMER         = "Latitude_summer"
    CLOUDS_SUMMER           = "Clouds_summer"
    CLOUDS_SNOW             = "Clouds_snow"
    SNOW_COVER_SNOW         = "Snow_snow"
    SPLIT                   = "split"


class HardNegatives:
    ANCHOR_LOC_ID           = "anchor_loc_id"
    NEGATIVE_LOC_ID         = "negative_loc_id"
    DISTANCE_M              = "distance_m"
