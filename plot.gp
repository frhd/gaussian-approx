# gnuplot plot.gp
# plots output from vizga csv export
# usage: gnuplot plot.gp
#   or:  gnuplot -e "filename='myfile.csv'" plot.gp

if (!exists("filename")) filename = "output.csv"

set datafile separator ","
set key outside right

# plot 1: true vs estimated position
set terminal dumb 80 24
set title "Position: True vs Estimated"
set xlabel "time"
set ylabel "position"
plot filename using 2:3 with lines title "true x", \
     filename using 2:5 with points title "meas x", \
     filename using 2:7 with lines title "est x"

# plot 2: estimation error
set title "Estimation Error (RMSE)"
set ylabel "rmse"
plot filename using 2:13 with lines title "rmse"

# plot 3: covariance trace
set title "Covariance Diagonal"
set ylabel "covariance"
plot filename using 2:11 with lines title "cov xx", \
     filename using 2:12 with lines title "cov yy"
