"""
Carbon Auditing Module for GreenBit
Tracks and calculates CO2 emissions from AI inference and storage
"""

from codecarbon import EmissionsTracker
from datetime import datetime
import json

class CarbonAudit:
    """Track and calculate carbon emissions"""
    
    def __init__(self, country_code="IN"):
        """
        Initialize carbon auditing
        
        Args:
            country_code: ISO country code for carbon intensity calculation
        """
        self.country_code = country_code
        self.tracker = None
        self.emissions_data = {}
        print(f"🌍 Carbon auditing initialized for {country_code}")
    
    def start_tracking(self):
        """Start carbon emissions tracking"""
        print("\n📊 Starting carbon emissions tracking...")
        
        try:
            self.tracker = EmissionsTracker(
                log_level="critical",
                measure_power_secs=5
            )
            self.tracker.start()
            self.start_time = datetime.now()
            print("✅ Carbon tracking started")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Could not start tracker: {e}")
            print("   Continuing without real-time tracking")
            return False
    
    def stop_tracking(self):
        """Stop tracking and get emissions"""
        if self.tracker:
            try:
                emissions = self.tracker.stop()
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                
                self.emissions_data = {
                    "timestamp": datetime.now().isoformat(),
                    "emissions_kg_co2": float(emissions) if emissions else 0.0,
                    "duration_seconds": elapsed_time,
                    "energy_consumed_kwh": float(emissions / 0.475) if emissions else 0.0,
                    "status": "tracked"
                }
            except Exception as e:
                print(f"⚠️ Warning: Could not stop tracker: {e}")
                self.emissions_data = {
                    "timestamp": datetime.now().isoformat(),
                    "emissions_kg_co2": 0.0,
                    "duration_seconds": 0,
                    "energy_consumed_kwh": 0.0,
                    "status": "estimated"
                }
        else:
            self.emissions_data = {
                "timestamp": datetime.now().isoformat(),
                "emissions_kg_co2": 0.0,
                "duration_seconds": 0,
                "energy_consumed_kwh": 0.0,
                "status": "not_tracked"
            }
        
        return self.emissions_data
    
    def calculate_storage_emissions(self, storage_gb, years=1, carbon_intensity=0.475):
        """
        Calculate CO2 from storage usage
        
        Args:
            storage_gb: Amount of storage in GB
            years: Duration storage is kept in years
            carbon_intensity: kg CO2 per kWh (varies by region)
            
        Returns:
            CO2 emissions in kg
        """
        # Estimates:
        # 1 GB stored for 1 year ≈ 0.00002 kg CO2 (0.2 grams)
        # Accounts for: storage equipment, cooling, electricity
        
        emission_factor = 0.00002  # kg CO2 per GB per year
        total_emissions = storage_gb * years * emission_factor
        
        return float(total_emissions)
    
    def calculate_net_environmental_gain(self, 
                                        storage_saved_gb, 
                                        inference_emissions_kg,
                                        duration_years=1):
        """
        Calculate environmental benefit of removing duplicates
        
        Args:
            storage_saved_gb: GB of storage freed
            inference_emissions_kg: CO2 from AI analysis
            duration_years: How long files would be kept
            
        Returns:
            Dictionary with environmental metrics
        """
        # Storage that would be kept if duplicates weren't removed
        storage_emissions_saved = self.calculate_storage_emissions(
            storage_saved_gb, 
            duration_years
        )
        
        net_gain = storage_emissions_saved - inference_emissions_kg
        
        if inference_emissions_kg > 0:
            roi = (net_gain / inference_emissions_kg) * 100
        else:
            roi = float('inf')
        
        return {
            "storage_freed_gb": float(storage_saved_gb),
            "storage_emissions_saved_kg": float(storage_emissions_saved),
            "inference_emissions_kg": float(inference_emissions_kg),
            "net_environmental_gain_kg": float(net_gain),
            "roi_percentage": float(roi),
            "duration_years": duration_years,
            "co2_equivalent": {
                "trees_needed": float(net_gain / 21),  # 1 tree absorbs ~21kg CO2/year
                "miles_driven": float(net_gain / 0.000411),  # 1 mile = 0.000411 kg CO2
                "households_electricity": float(net_gain / 4.7)  # 1 household = 4.7 tons/year
            }
        }
    
    def estimate_analysis_emissions(self, file_count, duration_seconds=60):
        """
        Estimate CO2 emissions from file analysis
        
        Args:
            file_count: Number of files analyzed
            duration_seconds: Analysis duration
            
        Returns:
            Estimated CO2 in kg
        """
        # Rough estimation:
        # Modern ML inference: ~0.02 kg CO2 per 1000 files per minute
        
        emission_per_1000_files = 0.02 / 60  # Per second
        estimated_emissions = (file_count / 1000) * emission_per_1000_files * duration_seconds
        
        return float(estimated_emissions)
    
    def get_carbon_report(self):
        """Get current carbon tracking report"""
        return self.emissions_data
    
    def format_carbon_display(self, emissions_kg):
        """
        Format CO2 emissions for display
        
        Args:
            emissions_kg: CO2 in kilograms
            
        Returns:
            Formatted string
        """
        if emissions_kg < 0.001:
            return f"{emissions_kg * 1e6:.2f} mg"
        elif emissions_kg < 1:
            return f"{emissions_kg * 1000:.2f} g"
        elif emissions_kg < 1000:
            return f"{emissions_kg:.4f} kg"
        else:
            return f"{emissions_kg / 1000:.4f} tons"

# Test the carbon audit module
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CARBON AUDITING MODULE")
    print("=" * 70)
    
    audit = CarbonAudit(country_code="IN")
    
    # Test 1: Storage emissions
    print("\n🔬 Test 1: Storage Emissions Calculation")
    storage_saved = 10  # GB
    emissions = audit.calculate_storage_emissions(storage_saved, years=1)
    print(f"  Storage saved: {storage_saved} GB")
    print(f"  CO2 saved annually: {audit.format_carbon_display(emissions)}")
    
    # Test 2: Analysis emissions estimation
    print("\n🔬 Test 2: Analysis Emissions Estimation")
    file_count = 1000
    duration = 120  # seconds
    analysis_emissions = audit.estimate_analysis_emissions(file_count, duration)
    print(f"  Files analyzed: {file_count}")
    print(f"  Duration: {duration} seconds")
    print(f"  Estimated CO2: {audit.format_carbon_display(analysis_emissions)}")
    
    # Test 3: Net environmental gain
    print("\n🔬 Test 3: Net Environmental Gain Calculation")
    gain = audit.calculate_net_environmental_gain(
        storage_saved_gb=100,  # 100 GB
        inference_emissions_kg=analysis_emissions,
        duration_years=1
    )
    print(f"  Storage freed: {gain['storage_freed_gb']} GB")
    print(f"  CO2 saved from storage: {audit.format_carbon_display(gain['storage_emissions_saved_kg'])}")
    print(f"  CO2 used for analysis: {audit.format_carbon_display(gain['inference_emissions_kg'])}")
    print(f"  Net gain: {audit.format_carbon_display(gain['net_environmental_gain_kg'])}")
    print(f"  ROI: {gain['roi_percentage']:.0f}x")
    print(f"  Equivalent to: {gain['co2_equivalent']['trees_needed']:.1f} trees planted")
    
    # Test 4: Carbon tracking (if available)
    print("\n🔬 Test 4: Carbon Tracking")
    audit.start_tracking()
    
    # Simulate some work
    import time
    print("  Simulating analysis work...")
    total = 0
    for i in range(1000000):
        total += i
    
    report = audit.stop_tracking()
    print(f"  Emissions tracked: {audit.format_carbon_display(report['emissions_kg_co2'])}")
    print(f"  Duration: {report['duration_seconds']:.2f} seconds")
    print(f"  Status: {report['status']}")
    
    print("\n✅ Carbon Auditing Module test complete!")
    print("=" * 70)
