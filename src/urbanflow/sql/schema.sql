CREATE TABLE IF NOT EXISTS taxi_zones (
    zone_id SMALLINT PRIMARY KEY CHECK (zone_id BETWEEN 1 AND 265),
    borough TEXT NOT NULL,
    zone_name TEXT NOT NULL,
    service_zone TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hourly_demand (
    hour TIMESTAMPTZ NOT NULL,
    zone_id SMALLINT NOT NULL REFERENCES taxi_zones(zone_id),
    demand INTEGER NOT NULL CHECK (demand >= 0),
    PRIMARY KEY (hour, zone_id)
);
CREATE INDEX IF NOT EXISTS hourly_demand_zone_hour ON hourly_demand(zone_id, hour DESC);
