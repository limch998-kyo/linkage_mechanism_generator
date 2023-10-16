import pylinkage as pl

# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1, name="B")
# Close the loop
pin = pl.Pivot(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1, name="C"
)

# Linkage definition
my_linkage = pl.Linkage(
    joints=(crank, pin),
    order=(crank, pin),
    name="My four-bar linkage"
)

# Visualization
pl.show_linkage(my_linkage)