Using the check on qqp the 0.7 cutoff failed as on 13-handpicked examples it looked good but on 2000 real pairs about 4 in 10 of the hits are wrong
cutoff	hits	wrong-hit %	missed-duplicate %
0.60	1362	47.5%	1.2%
0.70	1124	40.4%	7.5%
0.80	811	32.3%	24.2%
0.85	614	26.9%	38.0%
0.90	398	22.6%	57.5%
0.95	173	15.0%	79.7%


only score cannot fix it therefore added for fixes 
i did the wordcheck if it helps Wrong hits dropped only a little, about 3–5 points. Missed duplicates went way up: at 0.70, from 7.5% to 41.4%.



I applied the crossencoder where it gets candidates from first pass a 0.6 but it came out wrong as the doble checker was trained for a diffrent questions