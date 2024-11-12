# Import necessary libraries and set backend
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering on macOS

import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import cv2

# Define NOAA data URL
url = 'https://www.ncei.noaa.gov/thredds/dodsC/noaa-global-temp-v5/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202212_c20230108T133308.nc'
xrds = xr.open_dataset(url)

# Define frame dimensions based on the figure size (width, height)
fig_width, fig_height = 16, 8
dpi = 100  # dots per inch
frame_width, frame_height = int(fig_width * dpi), int(fig_height * dpi)

# Initialize VideoWriter object with calculated dimensions
video_writer = cv2.VideoWriter('output_spin_vid_v3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (frame_width, frame_height))

# Generator function to rotate globe
def longitude_generator(start=-180, stop=180, step=6):
    current = -95
    while True:
        yield current
        current += step
        if current >= stop:
            current = start + (current - stop)

longitude_gen = longitude_generator()

# Function to create and capture frame for each year and month
def create_and_capture_frame(year, month):
    data_for_month = xrds.sel(time=f'{year}-{month:02d}')
    mean_anom_matrix = data_for_month['anom'].mean(dim='time')
    longitude = next(longitude_gen)

    # Set up the figure and map projection
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=longitude))
    transform = ccrs.PlateCarree()

    try:
        # Plot data
        vmin, vmax = -3, 3
        pcm = mean_anom_matrix.plot(ax=ax, transform=transform, vmin=vmin, vmax=vmax, cmap='seismic', add_colorbar=False)
        ax.coastlines()
        ax.set_global()

        # Create grey background outside data extent
        xmin, xmax, ymin, ymax = ax.get_extent(transform)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='grey', transform=transform, zorder=-1))

        # Add color bar and title
        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical')
        cbar.set_label('Anomaly (°C)')
        ax.set_title(f'{year} - {month:02d}')

        # Render the plot to an image buffer
        plt.draw()
        plt_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((frame_height, frame_width, 4))
        plt_img = plt_img[:, :, :3]  # Remove alpha channel (RGBA -> RGB)
        frame = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        # Write frame to video and close the plot
        video_writer.write(frame)

    except TypeError as e:
        print(f"Skipping frame for {year}-{month:02d} due to geometry error: {e}")

    finally:
        plt.close(fig)

# Main loop to iterate through years and months
for year in range(2000, 2021):
    for month in range(1, 13):
        create_and_capture_frame(year, month)
    print(f'Year {year} finished.')

# Release video writer
video_writer.release()
print(f'Finished creating video!')
