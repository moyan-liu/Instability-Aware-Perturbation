# Cloud Seeding Perturbation Logic

## Overview

This document describes the physically-consistent cloud seeding perturbation applied to the Aurora weather model. The perturbation simulates glaciogenic cloud seeding by injecting ice-nucleating particles into supercooled liquid clouds, inducing ice formation, latent heat release, and precipitation.

---

## Physical Basis

**Cloud Seeding Mechanism**: Glaciogenic seeding introduces ice nuclei (typically silver iodide, AgI) into supercooled clouds, causing supercooled liquid droplets to freeze. This process:

1. **Ice Nucleation**: Converts supercooled liquid water to ice
2. **Latent Heat Release**: Releases latent heat of fusion (warming effect)
3. **Precipitation**: Ice crystals grow and fall as precipitation
4. **Moisture Depletion**: Removes water vapor from the atmosphere

---

## Mathematical Framework

### 1. Ice Nucleation

When ice nuclei are introduced, a fraction of atmospheric water vapor freezes:

```
q_frozen = q_old × η_freeze × mask_spatial
```

**Parameters**:
- `q_old`: Original specific humidity [kg/kg]
- `η_freeze`: Freezing efficiency = 0.25 (25% of water vapor freezes)
- `mask_spatial`: Spatial mask defining seeding region

**Physical Justification**:
- Typical cloud seeding achieves 10-30% conversion efficiency
- 25% represents moderately effective seeding under ideal conditions

---

### 2. Precipitation Formation

Not all frozen water precipitates immediately; some remains suspended as ice crystals:

```
q_precip = q_frozen × η_fallout
```

**Parameters**:
- `η_fallout`: Precipitation fallout fraction = 0.40 (40% falls as precipitation)
- Remaining 60% stays suspended temporarily

**Physical Justification**:
- Small ice crystals (< 100 μm) remain suspended by updrafts
- Larger crystals/aggregates fall as precipitation
- 40% fallout is typical for moderate convective conditions

---

### 3. Energy Balance

The thermodynamic energy change has TWO components:

#### 3a. Freezing Energy (Warming)
```
E_freeze = L_f × q_frozen
```
- All frozen water releases latent heat of fusion
- L_f = 334 kJ/kg (latent heat of fusion)

#### 3b. Precipitation Energy (Cooling)
```
E_precip = (L_v + L_f) × q_precip
```
- Precipitating water removes BOTH latent heats
- L_v = 2500 kJ/kg (latent heat of vaporization)
- Precipitation takes energy out of the atmospheric column

#### 3c. Net Energy Change
```
ΔE = L_f × q_frozen - (L_v + L_f) × q_precip
```

**Substituting** `q_precip = q_frozen × η_fallout`:
```
ΔE = L_f × q_frozen - (L_v + L_f) × (q_frozen × η_fallout)
ΔE = q_frozen × [L_f - (L_v + L_f) × η_fallout]
ΔE = q_frozen × [L_f × (1 - η_fallout) - L_v × η_fallout]
```

**With η_fallout = 0.40**:
```
ΔE = q_frozen × [334 × 0.60 - 2500 × 0.40]
ΔE = q_frozen × [200.4 - 1000]
ΔE = q_frozen × (-799.6 kJ/kg)
```

**Result**: NET COOLING because precipitation removes more energy than freezing releases.

---

### 4. Temperature Change

The energy change converts to temperature change via specific heat:

```
ΔT = ΔE / C_p
```

**Parameters**:
- C_p = 1005 J/(kg·K) (specific heat of air at constant pressure)
- ΔE in J/kg (convert kJ to J by multiplying by 1000)

**Final Temperature**:
```
T_new = T_old + ΔT
```

---

### 5. Moisture Removal

**CRITICAL PHYSICAL CONSTRAINT**: Aurora's `q` variable represents **water vapor only**, NOT total water content.

When water freezes, it leaves the vapor phase entirely:

```
q_new = q_old - q_frozen
```

**Common Error**: Previous implementation used `q_new = q_old - q_precip`, which only removed precipitated water. This violated mass conservation by leaving "ghost vapor" in the system.

**Correct Logic**:
- ALL frozen water (100%) leaves the vapor phase
- Whether it precipitates (40%) or stays suspended (60%) is irrelevant to vapor phase
- Ice crystals are NOT water vapor

---

### 6. Thermodynamic Consistency Check

After modification, we verify physical plausibility using the Clausius-Clapeyron relation:

#### 6a. Saturation Specific Humidity

Calculate saturation vapor pressure using the Magnus formula:

```
e_s = 6.112 × exp[(17.67 × T_celsius) / (T_celsius + 243.5)]  [hPa]
```

Convert to Pascal and calculate saturation specific humidity:

```
e_s_Pa = e_s × 100
q_sat = (ε × e_s_Pa) / (P - 0.378 × e_s_Pa)
```

**Parameters**:
- ε = 0.622 (ratio of molecular weights: M_water/M_air = 18.015/28.97)
- P = atmospheric pressure [Pa]
- T_celsius = T - 273.15 [°C]

#### 6b. Relative Humidity

```
RH = q_new / q_sat(T_new, P)
```

**Physical Bounds**:
- **RH ≥ 0**: Cannot have negative humidity
- **RH ≤ 1.3**: Allow 30% supersaturation (common in upper atmosphere)
- **q ≥ 0**: Cannot have negative specific humidity

**If violations occur**:
- Indicates unphysical parameter choices
- May require adjusting `η_freeze` or `η_fallout`
- Code reports violations for diagnostics

---

## Implementation Parameters

### Seeding Region

**Spatial Extent**:
- Longitude: 134°E - 136°E
- Latitude: 36°N - 38°N
- Pressure: 600-800 hPa (mid-troposphere)

**Seeding Point**:
- (135°E, 37°N) - marked with gold star in visualizations

### Physical Constants

| Constant | Value | Units | Description |
|----------|-------|-------|-------------|
| L_f | 334,000 | J/kg | Latent heat of fusion |
| L_v | 2,500,000 | J/kg | Latent heat of vaporization |
| C_p | 1,005 | J/(kg·K) | Specific heat of air |
| ε | 0.622 | - | Molecular weight ratio |
| η_freeze | 0.25 | - | Freezing efficiency |
| η_fallout | 0.40 | - | Precipitation fraction |

### Expected Impacts

**At 700 hPa with typical q ~ 0.003 kg/kg**:

**Moisture Change**:
- q_frozen ≈ 0.003 × 0.25 = 0.00075 kg/kg
- q_new ≈ 0.00225 kg/kg (25% reduction)

**Temperature Change**:
- ΔE ≈ 0.00075 × (-799,600) ≈ -600 J/kg
- ΔT ≈ -600 / 1005 ≈ -0.6 K (cooling)

**Relative Humidity**:
- RH_old ≈ 70% (typical)
- RH_new ≈ 52% (drier after seeding)

---

## Code Implementation

### Main Function

```python
def apply_physically_consistent_cloud_seeding(
    batch,
    freeze_efficiency=0.25,
    fallout_fraction=0.40,
    seed_lon_range=(134, 136),
    seed_lat_range=(36, 38),
    seed_hpa_range=(600, 800)
):
    """
    Apply physically-consistent cloud seeding perturbation.

    Returns:
        batch_seeded: Modified batch with T and q adjusted
        diagnostics: Dictionary with before/after statistics
    """
```

### Processing Steps

1. **Extract variables** from batch:
   - T (temperature)
   - q (specific humidity)
   - pressure levels
   - lat/lon coordinates

2. **Create spatial mask**:
   - Identify grid points within seeding region
   - Apply to specified pressure levels

3. **Calculate ice nucleation**:
   - `q_frozen = q × η_freeze × mask`

4. **Calculate precipitation**:
   - `q_precip = q_frozen × η_fallout`

5. **Calculate energy balance**:
   - `ΔE = L_f × q_frozen - (L_v + L_f) × q_precip`

6. **Update temperature**:
   - `T_new = T_old + ΔE / C_p`

7. **Update moisture** (CRITICAL):
   - `q_new = q_old - q_frozen` (remove ALL frozen water)

8. **Verify thermodynamics**:
   - Calculate `q_sat(T_new, P)`
   - Check `RH = q_new / q_sat`
   - Verify 0 ≤ RH ≤ 1.3
   - Check q_new ≥ 0

9. **Report diagnostics**:
   - ΔT range and mean
   - Δq range and mean
   - RH before/after
   - Number of violations (if any)

---

## Physical Interpretation

### Why Net Cooling?

Despite freezing releasing heat (L_f), the net effect is **cooling** because:

1. **Freezing** releases: +334 kJ/kg
2. **Precipitation** removes: -(2500 + 334) = -2834 kJ/kg

With 40% precipitation:
- Energy gained: 334 kJ/kg (100% of frozen water)
- Energy lost: 2834 × 0.40 = 1134 kJ/kg (40% precipitates)
- **Net**: -800 kJ/kg (cooling)

### Why This Matters

Cloud seeding doesn't just "make it rain" - it fundamentally alters:

1. **Thermodynamic structure**: Temperature profile changes
2. **Moisture distribution**: Dehumidification of seeded layer
3. **Stability**: Cooling can suppress further convection
4. **Dynamics**: Temperature/pressure changes alter wind patterns

The visualization shows how these thermodynamic changes propagate into **wind pattern modifications** - the ultimate goal of this perturbation study.

---

## Diagnostics Output

When running the perturbation, expect output like:

```
📋 PHYSICAL EQUATIONS APPLIED:
   1. Ice nucleation:        q_frozen = q × η_freeze × mask
   2. Precipitation:         q_precip = q_frozen × η_fallout
   3. Energy balance:        ΔE = L_f×q_frozen - (L_v+L_f)×q_precip
   4. Temperature change:    ΔT = ΔE / C_p
   5. Moisture removal:      q_new = q_old - q_frozen
   6. Saturation check:      RH = q_new / q_sat(T_new, P)

🌡️  Temperature Changes:
   Range: -0.82 K to -0.41 K
   Mean:  -0.59 K

💧 Moisture Changes:
   Range: -0.00091 to -0.00053 kg/kg
   Mean:  -0.00074 kg/kg

🌫️  Relative Humidity:
   Before: 68.3% (mean)
   After:  51.7% (mean)

✅ Thermodynamic checks: PASSED
   All RH values within bounds [0, 1.3]
   No negative q values detected
```

---

## References

**Physical Constants**:
- Wallace & Hobbs (2006): *Atmospheric Science: An Introductory Survey*
- Rogers & Yau (1989): *A Short Course in Cloud Physics*

**Cloud Seeding Physics**:
- Bruintjes (1999): "A review of cloud seeding experiments to enhance precipitation and some new prospects", *Bull. Amer. Meteor. Soc.*
- French et al. (2018): "Precipitation formation from orographic cloud seeding", *PNAS*

**Thermodynamics**:
- Bolton (1980): "The computation of equivalent potential temperature", *Mon. Wea. Rev.* (Magnus formula)
- Emanuel (1994): *Atmospheric Convection*

---

## Version History

- **v1.0** (2025-11-06): Initial implementation with physically-consistent T-q coupling
  - Fixed critical bug: Changed from `q_removed = q_precip` to `q_removed = q_frozen`
  - Added Clausius-Clapeyron thermodynamic consistency checks
  - Documented complete energy balance equations

---

**Author**: Cloud Seeding Perturbation Analysis
**Model**: Aurora (Microsoft/Arora-0.25-Pretrained)
**Purpose**: Quantify downstream dynamical impacts of cloud seeding on wind patterns
