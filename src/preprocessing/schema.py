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
    MAX_CLOUD_COVER         = 50  # percent


class Pairs:
    LOC_ID                  = "loc_id"
    INDEX_SUMMER            = "Index_summer"
    INDEX_SNOW              = "Index_snow"
    PATH_SUMMER             = "Path_summer"
    PATH_SNOW               = "Path_snow"
    CLOUDS_SUMMER           = "Clouds_summer"
    CLOUDS_SNOW             = "Clouds_snow"
    SNOW_COVER_SNOW         = "Snow_snow"
    SPLIT                   = "split"
