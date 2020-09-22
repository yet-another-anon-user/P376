import pyverdict
import pymysql
import pandas as pd
from tqdm import tqdm
import time
import sys
from utils import *

host = 'localhost';
port = 3306;
user = 'root';
password = '123456';

query_paths = {
    'taxi':'Taxi-RangeSumQuery-pickup_datetime-trip_distance-2000.jpk',
    'insta':'Instacart-RangeSumQuery-product_id-reordered-2000.jpk',
    'intel':'IntelWireless-RangeSumQuery-itime-light-2000.jpk',
}

target_attr = {
    'taxi':'trip_distance',
    'insta':'reordered',
    'intel':'light',
}


def create(name):
    conn = pyverdict.mysql(
        host=host,
        password=password,
        user=user,
        port=port
    )
    print('Creating a scrambled table for %s...'%name)
    conn.sql('DROP ALL SCRAMBLE nyctaxi.%s' % name)
    print("Dropped.")
    start = time.time()

    # PA tree cached on disk 86K, round up to 1M
    # data size = 481M, round up to 500M
    # our cost: 500*0.005 + 1 = 3.5/500 = 0.007
    percent = 0.1
    sql = 'CREATE SCRAMBLE nyctaxi.%s_scramble FROM nyctaxi.%s RATIO %s' % (name, name, percent)
    print(sql)
    ret = conn.sql(sql)
    print(ret)
    duration = time.time() - start
    conn.close()
    print('Scrambled table for nyctaxi.%s has been created.' % name)
    print('Time Taken = {:f} s'.format(duration))


def init():
    conn = pymysql.connect(
        host=host,
        password=password,
        port=port,
        user=user,
    )
    cur = conn.cursor()
    cur.execute('SET GLOBAL query_cache_size=0')

    vconn = pyverdict.mysql(
        host=host,
        user=user,
        password=password,
        port=port
    )
    return conn, vconn

def solve_one(m_conn, v_conn, query, name):

    start = time.time()
    cur = m_conn.cursor()
    cur.execute('SELECT SUM(%s) FROM nyctaxi.%s WHERE %s' % (target_attr[name], name, query))
    duration = time.time() - start
    print('Time Taken w/ MySQL = {:f} s'.format(duration))

    gt = cur.fetchone()[0]

    start = time.time()
    df = v_conn.sql('SELECT SUM(%s) FROM nyctaxi.%s_scramble WHERE %s' % (target_attr[name], name, query))
    duration = time.time() - start
    print('Time Taken w/ VerdictDB = {:f} s'.format(duration))

    est = df.values[0][0]

    if est is None:
        return 1.0

    re = float(abs(gt-est)/gt)
    # print("RE: %.5f" % re)
    return re

def solve_queries(qname):
    m_conn, v_conn = init()
    results = {}

    qpath = query_paths[qname]
    queries = restore_obj(qpath)
    qe = []
    print("Solving ", qname, qpath)
    for q in tqdm(queries):
        predicates = q.predicates
        str_p = ''
        for k,v in predicates.items():
            if str_p != '':
                str_p += " AND "
            str_p += k + ">=" + str(v[0]) + " AND " + k + "<=" + str(v[1])

        qe.append(solve_one(m_conn, v_conn, str_p, qname))

    results[qname] = qe

    dump_json(results, 'verdict-%s.json' % qname)
    m_conn.close()
    v_conn.close()


if __name__ == '__main__':
    name = sys.argv[2]
    if sys.argv[1] == 'create':
        create(name)
    elif sys.argv[1] == 'run':
        solve_queries(sys.argv[2])
