set grid
set xtics 64
set xlabel "Nthreads"
set ylabel "Performance, MLU/s"
set xrange [230:1050]
plot [][0:] "performance-LDST-512.dat" u 1:2 i 0 w lp lt 1 lw 3 t "FP D3Q19 CF=8x8x8    LD+ST only",\
                                    "" u 1:3 i 0 w lp lt 1 lw 2 t "FP D3Q19 CF=8x8x8    (2BLOCKperSM ) LD+ST only",\
        "performance-Periodic-512.dat" u 1:2 i 0 w lp lt 2 lw 3 t "FP D3Q19 CF=8x8x8    Periocic BC",\
              "performance-BB-512.dat" u 1:2 i 0 w lp lt 3 lw 3 t "FP D3Q19 CF=8x8x8    BBack BC",\
           "performance-LDST-1000.dat" u 1:2 i 0 w lp lt 4 lw 3 t "FP D3Q19 CF=10x10x10 LD+ST only",\
       "performance-Periodic-1000.dat" u 1:2 i 0 w lp lt 5 lw 3 t "FP D3Q19 CF=10x10x10 Periodic BC",\
             "performance-BB-1000.dat" u 1:2 i 0 w lp lt 6 lw 3 t "FP D3Q19 CF=10x10x10 BBack BC",\
      5908.21/1*512/169 lt 1 t "Peak 8x8x8 FP D3Q19",\
      5908.21/1*1000/271 lt 4 t "Peak 10x10x10 FP D3Q19"

pause -1;


plot [][0:] "performance-LDST-512.dat" u 1:2 i 1 w lp lt 1 lw 3 t "DP D3Q19 CF=8x8x8 LD+ST only",\
        "performance-Periodic-512.dat" u 1:2 i 1 w lp lt 2 lw 3 t "DP D3Q19 CF=8x8x8 Periocic BC",\
              "performance-BB-512.dat" u 1:2 i 1 w lp lt 3 lw 3 t "DP D3Q19 CF=8x8x8 BBack BC",\
      5908.21/2*512/169 lt 2 t "Peak 8x8x8 DP D3Q19"

pause -1;

plot [][0:] "performance-LDST-512.dat" u 1:2 i 2 w lp lt 1 lw 3 t "FP D3Q27 CF=8x8x8 LD+ST only",\
        "performance-Periodic-512.dat" u 1:2 i 2 w lp lt 2 lw 3 t "FP D3Q27 CF=8x8x8 Periocic BC",\
              "performance-BB-512.dat" u 1:2 i 2 w lp lt 3 lw 3 t "FP D3Q27 CF=8x8x8 BBack BC",\
      4157.63/1*512/169 lt 2 t "Peak 8x8x8 FP D3Q27"

pause -1
