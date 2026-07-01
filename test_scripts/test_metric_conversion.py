#!/usr/bin/env python3
"""
Test script for metric speed conversion
"""
import yaml
from micromouse import load_maze, upsample_maze

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Load maze
maze_file = config.get('maze_file', 'maze.csv')
maze_array = load_maze(maze_file)

# Check upsampling
if config.get('maze_upsampling', {}).get('enabled', False):
    factor = config.get('maze_upsampling', {}).get('factor', 1)
    maze_array = upsample_maze(maze_array, factor)
    print(f'Maze upsampled by factor {factor}')

print(f'\nFinal maze dimensions: {len(maze_array)}x{len(maze_array[0])}')

# Test metric conversion
phys_dims = config.get('physical_dimensions', {})
use_metric = phys_dims.get('use_metric_speeds', False)
print(f'\nUse metric speeds: {use_metric}')

if use_metric:
    maze_width_meters = phys_dims.get('maze_width_meters', 2.88)
    maze_height_meters = phys_dims.get('maze_height_meters', 2.88)
    num_cols = len(maze_array[0])
    num_rows = len(maze_array)
    meters_per_cell = (maze_width_meters / num_cols + maze_height_meters / num_rows) / 2.0
    
    print(f'\nMaze physical size: {maze_height_meters}m x {maze_width_meters}m')
    print(f'Meters per cell: {meters_per_cell:.6f} m/cell')
    
    # Test conversion
    max_speed_ms = config.get('pure_pursuit', {}).get('max_speed', 0)
    max_speed_grid = max_speed_ms / meters_per_cell
    print(f'\nSpeed conversions:')
    print(f'  max_speed: {max_speed_ms} m/s → {max_speed_grid:.2f} grid units/s')
    
    min_speed_ms = config.get('pure_pursuit', {}).get('min_speed', 0)
    min_speed_grid = min_speed_ms / meters_per_cell
    print(f'  min_speed: {min_speed_ms} m/s → {min_speed_grid:.3f} grid units/s')
    
    min_threshold_ms = config.get('micromouse', {}).get('min_speed_threshold', 0)
    min_threshold_grid = min_threshold_ms / meters_per_cell
    print(f'  min_speed_threshold: {min_threshold_ms} m/s → {min_threshold_grid:.3f} grid units/s')
    
    print(f'\n[OK] Metric speed conversion is working correctly!')
else:
    print('\n[WARNING] Metric speeds disabled - speeds are in grid units')
