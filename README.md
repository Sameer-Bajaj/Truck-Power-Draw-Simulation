This code is a revised version of @bborlaug's code that allows for simulation of large-scale EV Truck Fleets and their aggregate power draw throughout the day. This code returns the total power draw of
all three fleets across a specified interval, along with figures that show the aggregate power draw aswell as each fleet types' power draw in a muliiline plot. Users are also able to select whether they
want unscaled or scaled data to fit a specific use case. They can also choose the resolution, bound, directory, number of samples, and fleet sizes for each fleet type.

All differences in code from @bborlaug's code is written entirely by Sameer Bajaj, including the entire data_aggregate and agg_plot functions, aswell as modifications to the generate_load_profiles function.
