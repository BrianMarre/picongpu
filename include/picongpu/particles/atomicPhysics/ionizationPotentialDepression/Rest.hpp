/** check for and apply single step of pressure ionization cascade
 *
 * @attention assumes that ipd-input fields are up to date
 * @attention invalidates ipd-input fields if at least one ionization electron has been spawned
 *
 * @attention must be called once for each step in a pressure ionization cascade
 *
 * @tparam T_AtomicPhysicsIonSpeciesList list of all species partaking as ion in IPDIonization in atomicPhysics
 *
 * @attention collective over all ion species
 */
template<typename T_AtomicPhysicsIonSpeciesList>
HINLINE static void applyIPDIonization(picongpu::MappingDesc const mappingDesc)
{
    using IPDImplementation = ? ;
    using ForEachIonSpeciesApplyIPDIonization = pmacc::meta::
        ForEach<T_AtomicPhysicsIPDIonSpeciesList, ApplyIPDIonization<boost::mpl::_1, IPDImplementation>>;

    ForEachIonSpeciesApplyIPDIonization{}(mappingDesc);
};

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/LocalIPDInputFields.hpp"

// stewart pyatt
// ipd input fields
auto& localDebyeLengthField
    = *dc.get<s_IPD::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>("LocalDebyeLengthField");
auto& localTemperatureEnergyField
    = *dc.get<s_IPD::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
        "LocalTemperatureEnergyField");
auto& localZStarField = *dc.get<s_IPD::localHelperFields::LocalZStarField<picongpu::MappingDesc>>("LocalZStarField");

localDebyeLengthField.getDeviceDataBox(), localTemperatureEnergyField.getDeviceDataBox(),
    localZStarField.getDeviceDataBox()
