from utils import *

paths = [
    'Taxi1D-RangeSumQuery-pickup_time-trip_distance-1000.jpk',
    'Taxi2D-RangeSumQuery-pickup_date_pickup_time-trip_distance-1000.jpk',
    'Taxi3D-RangeSumQuery-PULocationID_pickup_date_pickup_time-trip_distance-1000.jpk',
    'Taxi4D-RangeSumQuery-PULocationID_dropoff_date_pickup_date_pickup_time-trip_distance-1000.jpk',
    'Taxi5D-RangeSumQuery-PULocationID_dropoff_date_dropoff_time_pickup_date_pickup_time-trip_distance-1000.jpk'
]

for path in paths:
    queries = restore_obj(path)

    template = "SELECT SUM({t_attr}) FROM taxi WHERE {predicates};"

    for q in queries:
        # print(q)
        predicates = q.predicates
        str_p = ''
        for k,v in predicates.items():
            if str_p != '':
                str_p += " AND "

            str_p += k + ">=" + str(v[0]) + " AND " + k + "<=" + str(v[1])
        sql = template.format(t_attr = q.target_attr, predicates = str_p)
        print(sql)


