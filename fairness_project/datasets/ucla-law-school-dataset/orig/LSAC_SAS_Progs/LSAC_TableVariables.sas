filename in1 'F:\storage\sas_bob\lsac\bp_base.dat';
libname out2 'C:\SAS_Files2004c\lsac';
options nocenter pagesize=7000;


data lsac;
  infile in1  ;
  input #1 ID 1-5 sex 6 race 7 cluster 9 lsat 10-13 ugpa 14-16  zfygpa 18-22 DOB_yr 25-26 
  grad $ 28 zgpa 37-41 bar1 $ 47 bar1_yr 50-51 bar2 $ 52 bar2_yr 53-54  fulltime 72 fam_inc 128 ;

age=bar1_yr-dob_yr;

if sex=1 then gender='female';
if sex=2 then gender='male  ';

if fulltime=1 then parttime=0;
if fulltime=2 then parttime=1;

if sex=2 then male=1;
if sex=1 then male=0;

if race=1 or race=8 then race1='other';
if race=7 then race1='white';
if race=2 then race1='asian';
if race=3 then race1='black';
if 3<race<7 then race1='hisp ';

if race=1 or race=8 then race2='c other';
if race=7 then race2='b white';
if race=2 then race2='c other';
if race=3 then race2='a black';
if 3<race<7 then race2='c other';

if grad='O' then Dropout='YES'; else dropout='NO ';

if race=1 or race=8 then other=1; else other=0;
if race=2 then asian=1; else asian=0;
if race=3 then black=1; else black=0;
if race=1 then hisp=1; else hisp=0;


if bar1='F' then pass_bar=0;
if bar1='P' then pass_bar=1;

if grad='O' then Dropout='YES'; else dropout='NO ';

if bar1='P'                               then bar='a Passed 1st time';
if bar2='P' and bar1 ne 'P'               then bar='b Passed 2nd time';
if bar2='F' or (bar1='F' and bar2= ' ' )  then bar='c Failed         ';
if bar1=' ' and bar2=' ' and grad ne 'X'  then bar='d Never took bar ';
if grad='O' or grad='N'                   then bar='e non-Grad       ';


if bar1='F' or bar2='F' then pass_bar=0;
if bar1='P' or bar2='P' then pass_bar=1;

if ugpa gt 4 then ugpa=4;

if cluster=5 then tier=6;
if cluster=4 then tier=5;
if cluster=1 then tier=4;
if cluster=3 then tier=3;
if cluster=2 then tier=2;
if cluster=6 then tier=1;

if ugpa gt 0 and lsat gt 0 then do;
index6040=((lsat-10) *15.789473) + (ugpa*100); end;
if index6040 gt 1000 then index6040=1000;

if index6040 lt 400   then indxgrp='a under 400';
if 400<=index6040<460 then indxgrp='b 400-460  ';
if 460<=index6040<520 then indxgrp='c 460-520  ';
if 520<=index6040<580 then indxgrp='d 520-580  ';
if 580<=index6040<640 then indxgrp='e 580-640  ';
if 640<=index6040<700 then indxgrp='f 640-700  ';
if 700<=index6040     then indxgrp='g 700+     ';

if index6040 lt 400   then indxgrp2='a under 400';
if 400<=index6040<460 then indxgrp2='b 400-460  ';
if 460<=index6040<520 then indxgrp2='c 460-520  ';
if 520<=index6040<580 then indxgrp2='d 520-580  ';
if 580<=index6040<640 then indxgrp2='e 580-640  ';
if 640<=index6040<700 then indxgrp2='f 640-700  ';
if 700<=index6040<760 then indxgrp2='g 700-760  ';
if 760<=index6040<820 then indxgrp2='h 760-820  ';
if 820<=index6040     then indxgrp2='i 820+     ';

label
/* Raw variables with column numbers of raw data */ 
ID='columns 1-5 respondent identification number'
sex='column 6 gender'
race='column 7 race'
cluster='column 9 cluster'
lsat='columns 10-13 LSAT score'
ugpa='columns 14-16 undegraduate GPA'
zfygpa='columns 18-22 standardized 1st year GPA'
DOB_yr='columns 25-26 year of birth'
grad='column 28 graduation status'
zgpa='columns 37-41 standardized overall GPA'
bar1='column 47 outcome of 1st bar exam'
bar1_yr='columns 50-51 year of 1st bar exam'
bar2='column 52 outcome of final bar exam'
bar2_yr='columns 53-54 year of final bar exam'
fulltime='column 72 fulltime/part time status'
fam_inc='column 128 family income'

/* Created varables */
decile1='Deciles for 1st year GPA after deleting missing values on RACE and CLUSTER'
decile1b='Deciles for 1st year GPA after deleting missing values on RACE, CLUSTER and OVERALL GPA'
decile3='Deciles for OVERALL GPA after deleting missing values on RACE, CLUSTER, and 1st YEAR GPA'
age='approximate age at 1st bar exam'
gender='categorical gender'
parttime='dummy variable for part time status'
male='dummy variable for male'
race1='5 category race variable'
race2='3 category race variable'
dropout='categorical variable for dropping out of law school'
bar='categorical outcome of law school'
pass_bar='dummy variable whether respondent ever passed bar' 
tier='prestige ranking of the 6 clusters'
index6040='weighted index using 60% of LSAT and 40% UGPA'
indexgrp='7 category grouping of index6040' 
indexgrp='9 category grouping of index6040' 
;

proc sort; by id; run;

data dec1; set out2.decile1;
proc sort; by id; run;

data dec3; set out2.decile3;
proc sort; by id; run;

data out2.lsac; merge dec3 dec1 lsac; by id; run;
