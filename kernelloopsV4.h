	#if NUMINDEXEDDIM>=3
	for (loopRng[2]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 2]; loopRng[2]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 2]; loopRng[2]++)
	#endif
	#if NUMINDEXEDDIM>=4
	for (loopRng[3]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 3]; loopRng[3]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 3]; loopRng[3]++)
	#endif
	#if NUMINDEXEDDIM>=5
	for (loopRng[4]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 4]; loopRng[4]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 4]; loopRng[4]++)
	#endif
	#if NUMINDEXEDDIM>=6
	for (loopRng[5]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 5]; loopRng[5]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 5]; loopRng[5]++)
	#endif
	#if NUMINDEXEDDIM>=7
	for (loopRng[6]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 6]; loopRng[6]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 6]; loopRng[6]++)
	#endif
	#if NUMINDEXEDDIM>=8
	for (loopRng[7]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 7]; loopRng[7]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 7]; loopRng[7]++)
	#endif
	#if NUMINDEXEDDIM>=9
	for (loopRng[8]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 8]; loopRng[8]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 8]; loopRng[8]++)
	#endif
	#if NUMINDEXEDDIM>=10
	for (loopRng[9]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 9]; loopRng[9]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 9]; loopRng[9]++)
	#endif
	#if NUMINDEXEDDIM>=11
	for (loopRng[10]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 10]; loopRng[10]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 10]; loopRng[10]++)
	#endif
	#if NUMINDEXEDDIM>=12
	for (loopRng[11]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 11]; loopRng[11]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 11]; loopRng[11]++)
	#endif
	#if NUMINDEXEDDIM>=13
	for (loopRng[12]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 12]; loopRng[12]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 12]; loopRng[12]++)
	#endif
	#if NUMINDEXEDDIM>=14
	for (loopRng[13]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 13]; loopRng[13]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 13]; loopRng[13]++)
	#endif
	#if NUMINDEXEDDIM>=15
	for (loopRng[14]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 14]; loopRng[14]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 14]; loopRng[14]++)
	#endif
	#if NUMINDEXEDDIM>=16
	for (loopRng[15]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 15]; loopRng[15]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 15]; loopRng[15]++)
	#endif
	#if NUMINDEXEDDIM>=17
	for (loopRng[16]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 16]; loopRng[16]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 16]; loopRng[16]++)
	#endif
	#if NUMINDEXEDDIM>=18
	for (loopRng[17]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 17]; loopRng[17]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 17]; loopRng[17]++)
	#endif
	#if NUMINDEXEDDIM>=19
	for (loopRng[18]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 18]; loopRng[18]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 18]; loopRng[18]++)
	#endif
	#if NUMINDEXEDDIM>=20
	for (loopRng[19]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 19]; loopRng[19]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 19]; loopRng[19]++)
	#endif
	#if NUMINDEXEDDIM>=21
	for (loopRng[20]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 20]; loopRng[20]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 20]; loopRng[20]++)
	#endif
	#if NUMINDEXEDDIM>=22
	for (loopRng[21]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 21]; loopRng[21]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 21]; loopRng[21]++)
	#endif
	#if NUMINDEXEDDIM>=23
	for (loopRng[22]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 22]; loopRng[22]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 22]; loopRng[22]++)
	#endif
	#if NUMINDEXEDDIM>=24
	for (loopRng[23]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 23]; loopRng[23]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 23]; loopRng[23]++)
	#endif
	#if NUMINDEXEDDIM>=25
	for (loopRng[24]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 24]; loopRng[24]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 24]; loopRng[24]++)
	#endif
	#if NUMINDEXEDDIM>=26
	for (loopRng[25]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 25]; loopRng[25]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 25]; loopRng[25]++)
	#endif
	#if NUMINDEXEDDIM>=27
	for (loopRng[26]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 26]; loopRng[26]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 26]; loopRng[26]++)
	#endif
	#if NUMINDEXEDDIM>=28
	for (loopRng[27]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 27]; loopRng[27]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 27]; loopRng[27]++)
	#endif
	#if NUMINDEXEDDIM>=29
	for (loopRng[28]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 28]; loopRng[28]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 28]; loopRng[28]++)
	#endif
	#if NUMINDEXEDDIM>=30
	for (loopRng[29]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 29]; loopRng[29]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 29]; loopRng[29]++)
	#endif
	#if NUMINDEXEDDIM>=31
	for (loopRng[30]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 30]; loopRng[30]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 30]; loopRng[30]++)
	#endif
	#if NUMINDEXEDDIM>=32
	for (loopRng[31]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 31]; loopRng[31]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 31]; loopRng[31]++)
	#endif
	#if NUMINDEXEDDIM>=33
	for (loopRng[32]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 32]; loopRng[32]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 32]; loopRng[32]++)
	#endif
	#if NUMINDEXEDDIM>=34
	for (loopRng[33]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 33]; loopRng[33]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 33]; loopRng[33]++)
	#endif
	#if NUMINDEXEDDIM>=35
	for (loopRng[34]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 34]; loopRng[34]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 34]; loopRng[34]++)
	#endif
	#if NUMINDEXEDDIM>=36
	for (loopRng[35]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 35]; loopRng[35]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 35]; loopRng[35]++)
	#endif
	#if NUMINDEXEDDIM>=37
	for (loopRng[36]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 36]; loopRng[36]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 36]; loopRng[36]++)
	#endif
	#if NUMINDEXEDDIM>=38
	for (loopRng[37]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 37]; loopRng[37]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 37]; loopRng[37]++)
	#endif
	#if NUMINDEXEDDIM>=39
	for (loopRng[38]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 38]; loopRng[38]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 38]; loopRng[38]++)
	#endif
	#if NUMINDEXEDDIM>=40
	for (loopRng[39]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 39]; loopRng[39]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 39]; loopRng[39]++)
	#endif
	#if NUMINDEXEDDIM>=41
	for (loopRng[40]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 40]; loopRng[40]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 40]; loopRng[40]++)
	#endif
	#if NUMINDEXEDDIM>=42
	for (loopRng[41]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 41]; loopRng[41]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 41]; loopRng[41]++)
	#endif
	#if NUMINDEXEDDIM>=43
	for (loopRng[42]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 42]; loopRng[42]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 42]; loopRng[42]++)
	#endif
	#if NUMINDEXEDDIM>=44
	for (loopRng[43]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 43]; loopRng[43]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 43]; loopRng[43]++)
	#endif
	#if NUMINDEXEDDIM>=45
	for (loopRng[44]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 44]; loopRng[44]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 44]; loopRng[44]++)
	#endif
	#if NUMINDEXEDDIM>=46
	for (loopRng[45]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 45]; loopRng[45]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 45]; loopRng[45]++)
	#endif
	#if NUMINDEXEDDIM>=47
	for (loopRng[46]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 46]; loopRng[46]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 46]; loopRng[46]++)
	#endif
	#if NUMINDEXEDDIM>=48
	for (loopRng[47]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 47]; loopRng[47]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 47]; loopRng[47]++)
	#endif
	#if NUMINDEXEDDIM>=49
	for (loopRng[48]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 48]; loopRng[48]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 48]; loopRng[48]++)
	#endif
	#if NUMINDEXEDDIM>=50
	for (loopRng[49]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 49]; loopRng[49]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 49]; loopRng[49]++)
	#endif
	#if NUMINDEXEDDIM>=51
	for (loopRng[50]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 50]; loopRng[50]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 50]; loopRng[50]++)
	#endif
	#if NUMINDEXEDDIM>=52
	for (loopRng[51]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 51]; loopRng[51]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 51]; loopRng[51]++)
	#endif
	#if NUMINDEXEDDIM>=53
	for (loopRng[52]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 52]; loopRng[52]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 52]; loopRng[52]++)
	#endif
	#if NUMINDEXEDDIM>=54
	for (loopRng[53]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 53]; loopRng[53]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 53]; loopRng[53]++)
	#endif
	#if NUMINDEXEDDIM>=55
	for (loopRng[54]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 54]; loopRng[54]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 54]; loopRng[54]++)
	#endif
	#if NUMINDEXEDDIM>=56
	for (loopRng[55]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 55]; loopRng[55]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 55]; loopRng[55]++)
	#endif
	#if NUMINDEXEDDIM>=57
	for (loopRng[56]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 56]; loopRng[56]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 56]; loopRng[56]++)
	#endif
	#if NUMINDEXEDDIM>=58
	for (loopRng[57]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 57]; loopRng[57]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 57]; loopRng[57]++)
	#endif
	#if NUMINDEXEDDIM>=59
	for (loopRng[58]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 58]; loopRng[58]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 58]; loopRng[58]++)
	#endif
	#if NUMINDEXEDDIM>=60
	for (loopRng[59]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 59]; loopRng[59]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 59]; loopRng[59]++)
	#endif
	#if NUMINDEXEDDIM>=61
	for (loopRng[60]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 60]; loopRng[60]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 60]; loopRng[60]++)
	#endif
	#if NUMINDEXEDDIM>=62
	for (loopRng[61]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 61]; loopRng[61]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 61]; loopRng[61]++)
	#endif
	#if NUMINDEXEDDIM>=63
	for (loopRng[62]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 62]; loopRng[62]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 62]; loopRng[62]++)
	#endif
	#if NUMINDEXEDDIM>=64
	for (loopRng[63]=rangeFilteredCellIdsMin[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 63]; loopRng[63]<=rangeFilteredCellIdsMax[(threadIdx.x % BLOCKSIZE) * NUMINDEXEDDIM + 63]; loopRng[63]++)
	#endif	
