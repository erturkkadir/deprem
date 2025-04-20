/* SQL Manager for MySQL                              5.9.1.55826 */
/* -------------------------------------------------------------- */
/* Host     : 192.168.1.166                                       */
/* Port     : 3306                                                */
/* Database : deprem                                              */


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES 'utf8mb4' */;

SET FOREIGN_KEY_CHECKS=0;

DROP DATABASE IF EXISTS `deprem`;

CREATE DATABASE `deprem`
    CHARACTER SET 'utf8mb4'
    COLLATE 'utf8mb4_general_ci';

USE `deprem`;

/* Structure for the `letters` table : */

CREATE TABLE `letters` (
  `lt_id` INTEGER NOT NULL AUTO_INCREMENT,
  `lt_x` INTEGER NOT NULL DEFAULT 0,
  `lt_y` INTEGER NOT NULL,
  `lt_m` INTEGER NOT NULL DEFAULT 0,
  `lt_cnt` INTEGER DEFAULT NULL,
  PRIMARY KEY USING BTREE (`lt_id`),
  KEY `lt_idx1` USING BTREE (`lt_x`),
  KEY `lt_idx2` USING BTREE (`lt_y`)
) ENGINE=InnoDB
ROW_FORMAT=DYNAMIC CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_general_ci';

/* Structure for the `tmp` table : */

CREATE TABLE `tmp` (
  `us_id` INTEGER NOT NULL AUTO_INCREMENT,
  `us_code` VARCHAR(30) COLLATE utf8mb4_general_ci NOT NULL,
  `us_datetime` DATETIME NOT NULL,
  `us_year` INTEGER DEFAULT NULL,
  `us_month` INTEGER DEFAULT NULL,
  `us_x` INTEGER DEFAULT NULL,
  `us_y` INTEGER DEFAULT NULL,
  `us_d` INTEGER DEFAULT NULL,
  `us_m` INTEGER DEFAULT NULL,
  `us_t` INTEGER DEFAULT NULL,
  `us_place` VARCHAR(40) COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY USING BTREE (`us_id`)
) ENGINE=MyISAM
CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_general_ci';

/* Structure for the `tmp2` table : */

CREATE TABLE `tmp2` (
  `us_id` INTEGER NOT NULL AUTO_INCREMENT,
  `us_datetime` DATETIME NOT NULL,
  PRIMARY KEY USING BTREE (`us_id`)
) ENGINE=MyISAM
ROW_FORMAT=FIXED CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_general_ci';

/* Structure for the `usgs` table : */

CREATE TABLE `usgs` (
  `us_id` INTEGER NOT NULL AUTO_INCREMENT,
  `us_code` VARCHAR(30) COLLATE utf8mb4_general_ci NOT NULL,
  `us_date` DATE DEFAULT NULL,
  `us_time` TIME DEFAULT '00:00:00',
  `us_datetime` DATETIME DEFAULT NULL,
  `us_lat` DOUBLE DEFAULT NULL,
  `us_lon` DOUBLE DEFAULT NULL,
  `us_dep` DOUBLE DEFAULT NULL,
  `us_mag` DOUBLE DEFAULT NULL,
  `us_magtype` VARCHAR(10) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `us_type` VARCHAR(40) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `us_x` INTEGER DEFAULT NULL,
  `us_y` INTEGER DEFAULT NULL,
  `us_d` INTEGER DEFAULT NULL,
  `us_m` INTEGER DEFAULT NULL,
  `us_t` INTEGER DEFAULT NULL,
  `us_c` INTEGER DEFAULT NULL,
  `us_crdate` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  `us_place` VARCHAR(100) COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY USING BTREE (`us_id`),
  UNIQUE KEY `us_code` USING BTREE (`us_code`),
  KEY `us_datetime` USING BTREE (`us_datetime`)
) ENGINE=InnoDB
ROW_FORMAT=DYNAMIC CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_general_ci';

/* Structure for the `usgs_tmp` table : */

CREATE TABLE `usgs_tmp` (
  `us_id` INTEGER NOT NULL AUTO_INCREMENT,
  `us_code` VARCHAR(30) COLLATE utf8mb4_general_ci NOT NULL,
  `us_date` DATE DEFAULT NULL,
  `us_time` TIME DEFAULT '00:00:00',
  `us_datetime` DATETIME DEFAULT NULL,
  `us_lat` DOUBLE DEFAULT NULL,
  `us_lon` DOUBLE DEFAULT NULL,
  `us_dep` DOUBLE DEFAULT NULL,
  `us_mag` DOUBLE DEFAULT NULL,
  `us_magtype` VARCHAR(10) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `us_type` VARCHAR(40) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `us_x` INTEGER DEFAULT NULL,
  `us_y` INTEGER DEFAULT NULL,
  `us_d` INTEGER DEFAULT NULL,
  `us_m` INTEGER DEFAULT NULL,
  `us_t` INTEGER DEFAULT NULL,
  `us_c` INTEGER DEFAULT NULL,
  `us_crdate` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  `us_place` VARCHAR(100) COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY USING BTREE (`us_id`),
  UNIQUE KEY `us_code` USING BTREE (`us_code`),
  KEY `us_datetime` USING BTREE (`us_datetime`)
) ENGINE=InnoDB
ROW_FORMAT=DYNAMIC CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_general_ci';


/* Definition for the `calc_dt` procedure : */

DELIMITER $$

CREATE DEFINER = 'kadir'@'%' PROCEDURE `calc_dt`(
        IN `usgs_id` INTEGER
    )
    NOT DETERMINISTIC
    CONTAINS SQL
    SQL SECURITY DEFINER
    COMMENT ''
BEGIN

DECLARE id INT;
DECLARE dt DATETIME;

select us_id, us_datetime into id, dt from usgs where us_datetime <= (select us_datetime from usgs where us_id = usgs_id) 
and us_id != usgs_id
order by us_datetime desc limit 1;

UPDATE usgs
SET us_t = -coalesce(TIMESTAMPDIFF(SECOND, us_datetime, dt), 0)
WHERE us_id = usgs_id;


END$$

DELIMITER ;

/* Definition for the `calc_letters` procedure : */

DELIMITER $$

CREATE DEFINER = 'kadir'@'%' PROCEDURE `calc_letters`()
    NOT DETERMINISTIC
    CONTAINS SQL
    SQL SECURITY DEFINER
    COMMENT ''
BEGIN

DROP TABLE IF EXISTS letters;
CREATE TABLE `letters` (
  `lt_id` INTEGER NOT NULL AUTO_INCREMENT,
  `lt_x` INTEGER NOT NULL DEFAULT 0,
  `lt_y` INTEGER NOT NULL,
  `lt_m` INTEGER NOT NULL DEFAULT 0,
  `lt_cnt` INTEGER DEFAULT NULL,
  PRIMARY KEY USING BTREE (`lt_id`),
  KEY `lt_idx1` USING BTREE (`lt_x`),
  KEY `lt_idx2` USING BTREE (`lt_y`)
) ENGINE=InnoDB
ROW_FORMAT=DYNAMIC CHARACTER SET 'utf8mb4' COLLATE 'utf8mb4_general_ci';

insert into letters(lt_x, lt_y, lt_m, lt_cnt)
select 
	us_x, us_y, us_m, count(*) from usgs 
where
	us_mag>3.99 and us_type = "earthquake" and us_magtype like 'm%'
group by 
    us_x, us_y, us_m; 


update 
  usgs u, letters l 
set 
  u.us_c = l.lt_id
where 
u.us_mag > 3.99 and 
u.us_x = l.lt_x and 
u.us_y = l.lt_y and 
u.us_m = l.lt_m; 
END$$

DELIMITER ;

/* Definition for the `get_data` procedure : */

DELIMITER $$

CREATE DEFINER = 'kadir'@'%' PROCEDURE `get_data`(
        IN `prMag` FLOAT
    )
    NOT DETERMINISTIC
    CONTAINS SQL
    SQL SECURITY DEFINER
    COMMENT ''
BEGIN

	DROP TABLE IF EXISTS `tmp`;
	CREATE TABLE `tmp` (
		`us_id` INTEGER NOT NULL AUTO_INCREMENT,
  		`us_code` VARCHAR(30) COLLATE utf8mb4_general_ci NOT NULL,
        `us_datetime` DATETIME NOT NULL,
        `us_year` INTEGER DEFAULT NULL,
        `us_month` INTEGER DEFAULT NULL,
  		`us_x` INTEGER DEFAULT NULL,
  		`us_y` INTEGER DEFAULT NULL,
  		`us_d` INTEGER DEFAULT NULL,
  		`us_m` INTEGER DEFAULT NULL,
  		`us_t` INTEGER DEFAULT NULL,
        `us_place` VARCHAR(40),
  		PRIMARY KEY USING BTREE (`us_id`)
	) ENGINE = MyISAM ROW_FORMAT=DEFAULT;
        
	INSERT INTO `tmp`(us_code, us_datetime, us_year, us_month, us_x, us_y, us_d, us_m, us_t, us_place)
	SELECT 
    	us_code, 
        us_datetime, 
    	year(us_datetime)-1970 as year, 
        month(us_datetime) as mont, 
        us_x,
        us_y,         
        us_d, 
        cast( (us_mag*10) as signed) as us_m, 
        0 as us_t,
        us_place
    FROM 
    	usgs 
    WHERE
		us_mag > prMag and 
        us_type = 'earthquake' and 
        us_magtype like 'm%' 
    order by  
    	us_datetime asc;    
        
        
    DROP TABLE IF EXISTS `tmp2`;
	CREATE TABLE `tmp2` (
		`us_id` INTEGER NOT NULL AUTO_INCREMENT,
        `us_datetime` DATETIME NOT NULL,
  		PRIMARY KEY USING BTREE (`us_id`)
	) ENGINE = MyISAM ROW_FORMAT=DEFAULT;
    insert into `tmp2` select us_id, us_datetime from tmp; 
    

	UPDATE `tmp` t1, `tmp2` t2
    	set t1.us_t =  -coalesce(TIMESTAMPDIFF(MINUTE, t1.`us_datetime`, t2.`us_datetime`),0)
	WHERE t1.us_id = (t2.us_id+1);
    
    select 
    	year(us_datetime)-1970 as year,
    	month(us_datetime) as mont,
    	us_x,
    	us_y,
    	us_m,
    	us_d,
		case when us_t>720 then 720 else cast(us_t as SIGNED) end as us_t    
     from tmp;
    
   
END$$

DELIMITER ;

/* Definition for the `ins_quakes` procedure : */

DELIMITER $$

CREATE DEFINER = 'kadir'@'%' PROCEDURE `ins_quakes`()
    NOT DETERMINISTIC
    CONTAINS SQL
    SQL SECURITY DEFINER
    COMMENT ''
BEGIN
insert into usgs(us_code, us_date, us_time, us_datetime, us_lat, us_lon, us_dep, us_mag, us_type,us_x, 
us_y, us_d, us_m, us_t, us_c, us_crdate, us_magtype, us_place) 
select us_code, us_date, us_time, us_datetime, us_lat, us_lon, us_dep, us_mag, us_type,  
us_x, us_y, us_d, us_m, us_t, us_c, us_crdate, us_magtype, us_place 
from usgs_tmp where us_code not in (select us_code from usgs) order by us_datetime asc;
END$$

DELIMITER ;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;