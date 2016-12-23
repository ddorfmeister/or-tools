jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	12		2 3 4 5 6 7 8 9 11 14 16 17 
2	3	5		32 28 21 18 10 
3	3	7		35 32 30 23 22 21 13 
4	3	8		35 32 27 23 22 21 19 15 
5	3	7		36 29 27 25 22 20 12 
6	3	9		35 34 33 32 30 26 22 21 18 
7	3	10		51 36 35 34 33 32 29 22 21 20 
8	3	10		35 33 32 30 29 27 25 23 22 21 
9	3	11		49 36 35 34 33 32 29 25 24 23 22 
10	3	9		51 49 35 34 33 29 24 22 20 
11	3	2		36 12 
12	3	8		51 48 35 34 33 32 26 21 
13	3	3		34 33 18 
14	3	8		50 49 34 33 32 29 24 23 
15	3	7		49 36 33 29 26 25 24 
16	3	5		51 32 29 26 21 
17	3	7		51 50 49 36 35 32 24 
18	3	5		36 31 29 27 25 
19	3	6		49 34 31 29 26 25 
20	3	3		50 48 23 
21	3	3		50 49 24 
22	3	7		50 48 45 44 38 37 31 
23	3	3		37 31 26 
24	3	5		45 44 38 37 31 
25	3	9		51 48 46 45 44 43 42 39 37 
26	3	8		47 46 45 44 42 40 39 38 
27	3	8		50 49 45 44 42 41 40 38 
28	3	8		51 47 45 44 43 42 40 39 
29	3	6		48 47 42 41 38 37 
30	3	6		48 46 44 42 41 38 
31	3	5		47 46 42 40 39 
32	3	5		47 46 44 42 37 
33	3	5		45 43 42 41 40 
34	3	4		44 42 41 37 
35	3	2		41 40 
36	3	2		48 40 
37	3	1		40 
38	3	1		43 
39	3	1		41 
40	3	1		52 
41	3	1		52 
42	3	1		52 
43	3	1		52 
44	3	1		52 
45	3	1		52 
46	3	1		52 
47	3	1		52 
48	3	1		52 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	6	11	24	10	25	22	9	22	21	
	2	16	5	21	8	11	22	7	12	21	
	3	22	1	19	7	9	21	5	9	21	
3	1	3	28	22	19	18	21	12	15	30	
	2	21	27	18	9	10	18	9	15	25	
	3	25	24	12	5	7	8	5	13	22	
4	1	7	26	14	21	26	21	22	14	24	
	2	19	22	13	19	24	20	12	14	23	
	3	30	11	9	18	24	17	4	8	22	
5	1	8	14	25	12	15	13	20	18	20	
	2	9	14	24	12	14	10	15	15	19	
	3	26	9	18	12	13	6	11	7	16	
6	1	23	18	28	26	17	6	21	15	28	
	2	26	12	22	19	16	6	21	12	26	
	3	30	9	17	13	16	6	20	9	25	
7	1	15	22	11	13	8	25	9	22	15	
	2	16	15	8	12	5	19	9	16	15	
	3	17	5	4	11	5	13	7	6	15	
8	1	3	23	28	12	30	27	18	19	8	
	2	16	22	27	10	30	22	10	10	6	
	3	17	21	25	4	30	19	5	5	6	
9	1	3	21	19	2	23	29	17	14	5	
	2	22	18	9	2	20	22	14	10	4	
	3	24	16	8	2	9	22	12	8	4	
10	1	16	22	25	28	26	15	20	25	23	
	2	18	19	14	28	16	12	18	23	19	
	3	22	11	6	28	10	11	5	12	17	
11	1	11	16	23	7	29	15	27	3	10	
	2	14	10	19	6	28	12	26	2	8	
	3	24	2	3	6	26	5	26	1	6	
12	1	11	14	11	16	28	18	28	3	25	
	2	16	10	8	12	21	14	16	2	19	
	3	24	9	6	12	14	13	11	2	18	
13	1	9	10	11	26	16	9	20	17	17	
	2	19	6	9	14	15	7	19	16	17	
	3	24	5	9	5	14	4	11	16	8	
14	1	12	25	13	27	26	22	6	1	22	
	2	23	25	13	24	20	21	3	1	15	
	3	25	25	13	22	16	21	2	1	6	
15	1	6	4	14	11	21	21	13	22	8	
	2	7	3	13	8	21	13	11	15	7	
	3	12	3	12	3	21	8	8	6	6	
16	1	2	20	18	6	26	24	17	16	24	
	2	6	11	9	6	22	15	12	11	21	
	3	12	4	2	6	16	11	12	7	19	
17	1	1	5	16	23	23	16	24	3	22	
	2	26	5	14	16	16	16	20	2	15	
	3	27	3	14	9	7	15	11	2	8	
18	1	2	22	26	26	2	20	24	26	10	
	2	3	16	24	24	1	18	21	24	7	
	3	4	12	20	23	1	13	19	21	6	
19	1	2	5	19	2	24	4	21	28	29	
	2	20	5	18	2	22	4	16	27	29	
	3	23	4	16	1	20	4	7	24	29	
20	1	5	24	16	3	20	11	18	10	19	
	2	18	21	14	3	16	11	15	6	19	
	3	30	12	8	3	12	9	15	5	19	
21	1	1	20	26	15	19	17	17	24	26	
	2	4	15	23	10	16	12	16	15	23	
	3	19	8	20	7	12	6	11	8	21	
22	1	8	28	7	17	30	21	10	19	24	
	2	12	23	6	9	27	21	10	16	21	
	3	17	19	6	8	21	18	8	7	21	
23	1	5	22	25	24	16	11	9	22	15	
	2	9	15	21	12	15	8	9	14	14	
	3	15	10	18	5	15	7	5	7	12	
24	1	1	10	26	9	23	26	19	22	29	
	2	9	7	25	6	20	20	13	21	21	
	3	13	7	22	3	17	12	7	19	15	
25	1	13	15	20	28	9	14	21	20	17	
	2	22	13	13	24	7	11	16	10	14	
	3	29	13	2	20	6	10	7	7	9	
26	1	3	24	13	21	22	29	24	28	22	
	2	4	18	7	20	22	25	21	25	20	
	3	21	5	3	10	22	22	20	23	19	
27	1	12	9	20	19	11	16	11	25	17	
	2	18	8	19	16	10	14	10	18	12	
	3	25	7	11	10	7	8	7	17	6	
28	1	8	21	25	24	17	23	24	19	21	
	2	15	11	19	17	16	12	21	15	19	
	3	30	8	10	13	15	7	12	12	1	
29	1	6	21	25	13	17	22	29	13	15	
	2	19	17	19	11	13	11	23	11	10	
	3	27	10	15	5	7	1	12	9	8	
30	1	13	30	26	26	18	16	25	21	15	
	2	27	21	22	22	12	14	22	16	11	
	3	29	14	21	20	9	13	17	12	9	
31	1	7	19	7	18	11	16	27	10	22	
	2	15	17	4	12	8	10	26	7	19	
	3	22	14	1	8	7	6	24	6	17	
32	1	2	25	11	25	23	29	21	15	19	
	2	5	23	11	24	23	29	20	13	9	
	3	19	22	10	24	22	29	20	10	3	
33	1	1	14	25	26	15	19	28	21	30	
	2	9	9	13	23	13	14	20	20	21	
	3	10	4	7	16	8	13	12	18	15	
34	1	7	24	22	17	29	11	19	7	9	
	2	11	21	19	16	28	8	15	5	9	
	3	23	15	18	10	27	7	13	5	4	
35	1	12	29	4	15	21	19	20	29	29	
	2	17	29	3	14	20	14	14	25	28	
	3	26	29	1	14	16	14	11	23	26	
36	1	8	12	30	28	12	16	18	26	23	
	2	10	7	23	25	10	9	14	19	22	
	3	21	4	15	20	10	7	14	13	19	
37	1	10	19	14	22	16	18	5	19	5	
	2	11	19	10	17	10	14	4	18	4	
	3	22	19	5	17	3	8	4	18	3	
38	1	6	30	24	13	23	4	20	25	17	
	2	16	30	23	13	19	4	16	23	11	
	3	21	30	22	11	12	3	9	19	10	
39	1	12	21	18	20	10	27	21	5	13	
	2	17	19	10	16	9	23	20	4	10	
	3	19	17	3	16	3	20	20	2	6	
40	1	5	22	14	27	22	22	18	18	12	
	2	6	22	8	26	22	16	13	16	8	
	3	18	21	8	22	21	5	5	16	6	
41	1	11	28	29	25	17	7	30	23	13	
	2	19	24	26	13	17	7	27	23	8	
	3	29	19	22	11	15	7	27	22	4	
42	1	1	9	12	16	21	28	5	15	17	
	2	16	9	11	13	13	13	4	14	9	
	3	28	9	5	11	11	11	3	6	8	
43	1	5	11	23	25	15	9	21	12	17	
	2	26	9	21	22	12	8	18	11	17	
	3	30	6	20	19	9	1	3	11	16	
44	1	2	20	21	26	20	11	21	17	23	
	2	7	15	13	24	13	8	20	13	21	
	3	23	6	6	22	13	7	19	3	17	
45	1	8	16	26	20	24	15	24	13	17	
	2	11	14	23	19	20	8	23	11	15	
	3	28	8	18	19	12	2	21	11	15	
46	1	5	15	29	18	14	20	27	25	21	
	2	13	14	19	15	10	18	24	25	19	
	3	21	13	12	14	10	16	24	18	17	
47	1	6	26	8	25	18	18	28	9	9	
	2	12	26	7	10	16	15	23	5	5	
	3	22	26	5	4	10	14	17	1	3	
48	1	9	20	26	18	23	9	21	16	23	
	2	23	15	18	16	22	9	13	15	23	
	3	30	7	6	15	20	7	9	15	22	
49	1	3	28	15	9	16	27	23	11	13	
	2	12	25	14	7	12	24	20	5	5	
	3	22	21	10	7	10	24	15	5	5	
50	1	2	4	20	27	27	22	20	22	23	
	2	27	3	17	26	21	16	16	21	21	
	3	30	1	14	26	19	10	6	17	15	
51	1	1	26	19	19	20	21	13	23	2	
	2	17	23	11	18	18	13	6	22	1	
	3	20	19	8	18	3	5	3	20	1	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	81	82	79	89	813	869	785	845

************************************************************************