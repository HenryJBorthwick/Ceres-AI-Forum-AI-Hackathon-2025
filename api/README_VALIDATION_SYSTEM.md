# IPC Data Validation System

## Overview

The IPC Data Validation System is a comprehensive solution that ensures only valid regions (those with both IPC historical data AND satellite embeddings) are displayed in the graph configurator. This system handles the complex mapping between IPC dataset region names and satellite embedding region names, accounting for naming inconsistencies and ensuring data quality.

## Problem Solved

The original system had several issues:

1. **Naming Inconsistencies**: IPC data and satellite embeddings use different naming conventions for the same regions
2. **Missing Data**: Some regions existed in IPC data but not in satellite embeddings (and vice versa)
3. **Invalid Selections**: Users could select regions that couldn't be used for AI inference
4. **Poor User Experience**: Graph configurator showed regions that would fail when trying to generate predictions

## Solution Architecture

### 1. Comprehensive Region Mapping (`comprehensive_region_mapping.py`)

- **Manual Mappings**: Pre-defined mappings for known naming discrepancies
- **Bidirectional Mapping**: Convert between IPC and satellite region names
- **Fallback Logic**: Returns original name if no mapping exists

### 2. Data Validator (`data_validator.py`)

- **Validation Engine**: Core class that validates data consistency
- **Caching System**: Performance-optimized validation with caching
- **Mapping Status**: Tracks exact matches, mapped regions, and unmapped regions
- **Coverage Analysis**: Provides comprehensive data coverage statistics

### 3. Enhanced Data Loader (`data_loader.py`)

- **Integration**: Integrates with validation system
- **Filtering**: Only returns valid regions for countries
- **Inference Validation**: Ensures regions can be used for AI predictions
- **Fallback Support**: Maintains backward compatibility

### 4. Updated API (`app.py`)

- **Validation Endpoints**: New endpoints for data validation
- **Filtered Responses**: Only returns valid countries and regions
- **Error Handling**: Graceful fallback if validation fails
- **Status Reporting**: Detailed inference status for countries

## Key Features

### ✅ Data Validation
- Ensures regions exist in both IPC and satellite datasets
- Handles naming inconsistencies automatically
- Provides detailed mapping status for each region

### ✅ Performance Optimization
- Caches validation results for fast responses
- Efficient region filtering algorithms
- Minimal impact on API response times

### ✅ User Experience
- Only shows regions that can be used for inference
- Clear indication of AI prediction availability
- Prevents selection of invalid regions

### ✅ Maintainability
- Comprehensive logging and error handling
- Easy to add new region mappings
- Automated testing and validation scripts

## Usage

### Basic Usage

```python
from data_validator import DataValidator
from data_loader import load_data, load_satellite_embeddings_data

# Load data
ipc_data = load_data()
satellite_data = load_satellite_embeddings_data()

# Create validator
validator = DataValidator(ipc_data, satellite_data)

# Get valid countries
valid_countries = validator.get_valid_countries()

# Get valid regions for a country
valid_regions = validator.get_valid_regions_for_country('AFG')

# Check if a region can be used for inference
can_infer = validator.validate_region_for_inference('AFG', 'Badakhshan')
```

### API Endpoints

#### Get Valid Countries
```http
GET /countries
```
Returns only countries that have both IPC data and satellite embeddings.

#### Get Valid Regions
```http
GET /levels/{country}
```
Returns only regions that exist in both datasets for the specified country.

#### Data Validation Report
```http
GET /data-validation-report
```
Provides comprehensive mapping statistics and coverage information.

## Adding New Region Mappings

### 1. Manual Mapping

Add new mappings to `comprehensive_region_mapping.py`:

```python
REGION_MAPPING = {
    # Existing mappings...
    
    # New mapping
    'IPC_Region_Name': 'Satellite_Region_Name',
}
```

### 2. Automatic Detection

Run the mapping analysis script to identify new regions that need mapping:

```bash
cd api
python analyze_mapping_gaps.py
```

This will generate:
- `mapping_analysis_report.txt`: Comprehensive analysis
- `potential_mappings.txt`: Suggested mappings based on similarity

### 3. Validation

Test new mappings with the validation system:

```bash
python test_validation_system.py
```

## Testing

### Run All Tests
```bash
cd api
python test_validation_system.py
```

### Run Mapping Analysis
```bash
python analyze_mapping_gaps.py
```

### Test API Endpoints
```bash
# Start the API server
uvicorn app:app --reload

# Test endpoints
curl http://localhost:8000/countries
curl http://localhost:8000/levels/Afghanistan
curl http://localhost:8000/data-validation-report
```

## Monitoring and Maintenance

### Regular Tasks

1. **Review Unmapped Regions**: Check for new regions that need mapping
2. **Update Mappings**: Add new mappings as they are identified
3. **Monitor Coverage**: Track data coverage percentage over time
4. **Performance Monitoring**: Monitor validation system performance

### Reports Generated

- **Mapping Report**: Detailed mapping status for all regions
- **Coverage Summary**: Overall data coverage statistics
- **Unmapped Regions**: List of regions needing attention
- **Potential Mappings**: AI-suggested mappings for review

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Loading Failures**: Check file paths and data formats
3. **Validation Failures**: Review mapping configurations
4. **Performance Issues**: Check cache configuration and data size

### Debug Mode

Enable detailed logging in `app.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Fallback Mode

The system includes fallback mechanisms:
- If validation fails, falls back to original behavior
- Graceful degradation ensures API remains functional
- Error logging for debugging and monitoring

## Future Enhancements

### Planned Features

1. **Machine Learning Mappings**: AI-powered region name matching
2. **Real-time Validation**: Dynamic validation as data changes
3. **User Feedback**: Allow users to suggest new mappings
4. **Advanced Analytics**: Deeper insights into data quality

### Scalability

- **Distributed Caching**: Redis-based caching for large datasets
- **Async Processing**: Background validation for large datasets
- **Incremental Updates**: Efficient updates for changing data

## Contributing

### Adding New Mappings

1. Identify unmapped regions using analysis tools
2. Research correct mappings from authoritative sources
3. Add mappings to `comprehensive_region_mapping.py`
4. Test with validation system
5. Update documentation

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality
- Update this README for significant changes

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review generated reports for insights
3. Run validation tests to identify problems
4. Check logs for detailed error information

## License

This system is part of the Ceres AI IPC Prediction Platform project.
