Some notes for setting up the VerdictDB experiment.

psql -d nyctaxi -c "COPY taxi FROM '`pwd`/taxi6-noheader.csv' DELIMITER ',' ENCODING 'LATIN1';"

CREATE TABLE taxi (
  pickup_date numeric,
  pickup_datetime numeric,
  dropoff_datetime numeric,
  pickup_time numeric,
  dropoff_date numeric,
  dropoff_time numeric,
  PULocationID numeric,
  DOLocationID numeric,
  trip_distance numeric
);


CREATE TABLE insta (
  order_id numeric,
  product_id numeric,
  add_to_cart_order numeric,
  reordered numeric,
);


CREATE TABLE intel (
  date varchar(20),
  time varchar(20),
  epoch numeric,
  moteid numeric,
  temperature numeric,
  humidity numeric,
  light numeric,
  voltage numeric,
  idate numeric,
  itime numeric
);




follow this guide *exactly* to get an older version mysql running w/o sudo: https://dzone.com/articles/setting-up-mysql-without-root-access
# to init
./mysqld --datadir=$DATADIR --basedir=$MYSQL_HOME --log-error=$MYSQL_HOME/log/mysql.err --pid-file=$MYSQL_HOME/mysql.pid --socket=$MYSQL_HOME/socket --port=3306 --initialize-insecure
# to start after initialization
./mysqld --datadir=$DATADIR --basedir=$MYSQL_HOME --log-error=$MYSQL_HOME/log/mysql.err --pid-file=$MYSQL_HOME/mysql.pid --socket=$MYSQL_HOME/socket --port=3306
# to stop
./mysqld --datadir=$DATADIR --basedir=$MYSQL_HOME --log-error=$MYSQL_HOME/log/mysql.err --pid-file=$MYSQL_HOME/mysql.pid --socket=$MYSQL_HOME/socket --port=3306 stop
# to connect, find the password in the logfile then
./mysql -u root -h 127.0.0.1 -p
# reset pwd.
SET PASSWORD = PASSWORD('123456');

# load data similarly like this: https://docs.verdictdb.org/documentation/step_by_step_tutorial/tpch_load_data/
# in mysql shell
LOAD DATA LOCAL INFILE '/local/xi/VarAcc/data/taxi6-noheader.csv'     INTO TABLE taxi     FIELDS TERMINATED BY ',';

LOAD DATA LOCAL INFILE '/local/xi/VarAcc/data/intel_i.csv'     INTO TABLE intel     FIELDS TERMINATED BY ',';

LOAD DATA LOCAL INFILE '/local/xi/VarAcc/data/insta.csv'     INTO TABLE insta     FIELDS TERMINATED BY ',';

#measure the size ofthe table for a fair competition

SELECT
  TABLE_NAME AS `Table`,
  ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024) AS `Size (MB)`
FROM
  information_schema.TABLES
WHERE
  TABLE_SCHEMA = "nyctaxi"
ORDER BY
  (DATA_LENGTH + INDEX_LENGTH)
DESC;
