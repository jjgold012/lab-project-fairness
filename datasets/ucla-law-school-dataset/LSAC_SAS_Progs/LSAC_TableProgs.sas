libname in 'C:\SAS_Files2004c\lsac';
options pagesize=60 linesize=190 nocenter nodate;


data table31; set in.lsac;
if cluster=5;
if race1='white' or race1='black';
run;
Title1'data for Table 3.1';
proc means n mean median std maxdec=0;
class race1;
var index6040; run;

data table32; set in.lsac;
if race1='white' or race1='black';
run;
Title1'data for Table3.2';
proc means n median std maxdec=0;
class tier race1;
var index6040; run;

data table51; set in.lsac;
title1'Table 5.1 and part of Table 5.3 First-year zgpa';
Proc freq ;
tables tier*decile1*race2/norow nopercent;
run;

data lsac; set in.lsac;
t=1; 
if cluster ne ' ' and zfygpa ne .;
proc sort; by cluster;
run;

data Table53; set lsac;
if race1='black';
Title1'Blacks Ns for Table 5.3';
proc means median;
class cluster;
var zfygpa; 
output out=median median=blkmed;
run;

data median2; set median;
t=1;
keep t blkmed cluster;
run;

data m; merge lsac median2; by cluster t;
if race1='white';
if zfygpa ne .;
if zfygpa<=blkmed then whtper=1; else whtper=2; run;

Title1'Whites Percentiles for Table 5.3';
proc freq;
tables cluster*whtper/ nopercent nocol out=whiteper; run;

data wht; set whiteper;
proc sort; by cluster whtper; run;

proc transpose out=tran;
id whtper;
var count;
by cluster ; run;

data tran2; set tran;
percent=(_1/(_1 + _2))*100;
if cluster=5 then tier=6;
if cluster=4 then tier=5;
if cluster=1 then tier=4;
if cluster=3 then tier=3;
if cluster=2 then tier=2;
if cluster=6 then tier=1;

if tier=6 then group=1;
if tier=5 then group=2;
if tier=4 then group=3;
if tier=3 then group=4;
if tier=2 then group=5;
if tier=1 then group=6;
keep group percent;
proc sort; by group;
Title1'Whites Percentiles for Table 5.3';
proc print; run;

data table54; set in.lsac;
if race1='black';
proc sort; by race1;
proc freq ;
tables decile1b;
by race1;
run;
title1'3rd Year Data for Table5.4';
proc freq ;
tables decile3;
by race1;
run;

data Table55; set in.lsac;
Title1'data for Table5.5';
proc freq;
tables tier*bar*race1/; run;

title1'TABLE 5.6 ';
proc logistic  DESCENDING;
model dropout= zfygpa tier parttime fam_inc male black asian other hisp /rsq stb
 lackfit;
run;

data table57; set in.lsac;
if race1=' ' then delete ;
if race1='white' or race1='black';
if index6040 ne .;
 Title1'data for Table 5.7';
proc freq;
tables race1*indxgrp*bar/nopercent nocol; run;

data Table61; set in.lsac;

title1'Table 6.1 ';
proc logistic  DESCENDING;
model pass_bar=zgpa lsat tier ugpa male asian black other hisp /rsq stb
 lackfit;
run;

data table62; set in.lsac;
if race1='black' or race1='white';
if index6040 ne .;
Title1'data for Table6.2';
proc freq;
tables race1*indxgrp2*bar1/nopercent nocol; run;
