-- UTC storage, NYC business calendar. $1 and $2 are UTC interval bounds.
SELECT z.borough, z.zone_name,
       EXTRACT(HOUR FROM h.hour AT TIME ZONE 'America/New_York') AS local_hour,
       AVG(h.demand) AS mean_pickups, SUM(h.demand) AS total_pickups
FROM hourly_demand h JOIN taxi_zones z USING (zone_id)
WHERE h.hour >= $1 AND h.hour < $2
GROUP BY z.borough, z.zone_name, local_hour
ORDER BY total_pickups DESC;

-- Window ends before the target hour; feature generation never includes its label.
SELECT hour, zone_id, demand,
       LAG(demand, 24) OVER w AS lag_24,
       AVG(demand) OVER (PARTITION BY zone_id ORDER BY hour
                        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS mean_24
FROM hourly_demand
WINDOW w AS (PARTITION BY zone_id ORDER BY hour);
