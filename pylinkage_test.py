import pylinkage as pl

# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1)
# Close the loop
pin = pl.Pivot(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1
)

# Create the linkage
my_linkage = pl.Linkage(joints=(crank, pin))

# Show the results
pl.show_linkage(my_linkage)