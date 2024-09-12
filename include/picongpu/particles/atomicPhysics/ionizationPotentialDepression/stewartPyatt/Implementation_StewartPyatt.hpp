/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file Stewart-Pyatt ionization potential depression(IPD) implementation
 *
 * Implements the interface of stewart-pyatt like IPD models with the rest of atomicPhysics stage.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/ApplyIPDIonization.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/MatterIPDImplementation.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/CalculateIPDInput.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/FillIPDSumFields_Electron.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/FillIPDSumFields_Ion.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/LocalIPDInputFields.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/SumFields.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalFoundUnboundIonField.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics::ionizationPotentialDepression;

    /** implementation of Stewart-Pyatt ionization potential depression(IPD) model
     *
     * @tparam T_TemperatureFunctor term A to average over for all macro particles according to equi-partition theorem,
     *      must follow average(A) = k_B * T
     * @tparam T_IPDModel model to use in calculateIPD call
     */
    template<typename T_TemperatureFunctor, typename T_IPDModel>
    struct Implementation_StewartPyatt : MatterIPDImplementation
    {
    private:
        //! reset IPD support infrastructure before we accumulate over particles to calculate new IPD Inputs
        HINLINE static void resetSumFields()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localSumWeightAllField
                = *dc.get<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>("SumWeightAllField");
            localSumWeightAllField.getDeviceBuffer().setValue(0._X);

            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    "SumTemperatureFunctionalField");
            localSumTemperatureFunctionalField.getDeviceBuffer().setValue(0._X);

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    "SumWeightElectronsField");
            localSumWeightElectronField.getDeviceBuffer().setValue(0._X);

            auto& localSumChargeNumberIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberIonsField");
            localSumChargeNumberIonsField.getDeviceBuffer().setValue(0._X);

            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    "SumChargeNumberSquaredIonsField");
            localSumChargeNumberSquaredIonsField.getDeviceBuffer().setValue(0._X);
        }

    public:
        //! create the sum- and IPD-Input superCell fields required by Stewart-Pyatt
        HINLINE static void createHelperFields(
            picongpu::DataConnector& dataConnector,
            picongpu::MappingDesc const mappingDesc)
        {
            // create sumFields
            //@{
            auto sumWeightAllField
                = std::make_unique<s_IPD::localHelperFields::SumWeightAllField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(sumWeightAllField));
            auto sumTemperatureFunctionalField
                = std::make_unique<s_IPD::localHelperFields::SumTemperatureFunctionalField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumTemperatureFunctionalField));

            auto sumWeightElectronsField
                = std::make_unique<s_IPD::localHelperFields::SumWeightElectronsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumWeightElectronsField));

            auto sumChargeNumberIonsField
                = std::make_unique<s_IPD::localHelperFields::SumChargeNumberIonsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumChargeNumberIonsField));
            auto sumChargeNumberSquaredIonsField
                = std::make_unique<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(sumChargeNumberSquaredIonsField));
            //@}

            // create IPD input Fields
            //@{
            // in sim.unit.length(), not weighted
            auto localDebyeLengthField
                = std::make_unique<s_IPD::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(localDebyeLengthField));

            // z^star IPD input field, z^star = = average(q^2) / average(q) ;for q charge number of ion, unitless,
            //  not weighted
            auto localZStarField
                = std::make_unique<s_IPD::localHelperFields::LocalZStarField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(localZStarField));

            // local k_Boltzman * Temperature field, in eV
            auto localTemperatureEnergyField
                = std::make_unique<s_IPD::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(localTemperatureEnergyField));
            //@}
        }

        /** calculate all inputs for the ionization potential depression
         *
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
         *  atomicPhysics kernels if no atomic physics species is present.
         *
         * @attention collective over all IPD species
         */
        template<
            uint32_t T_numberAtomicPhysicsIonSpecies,
            typename T_IPDIonSpeciesList,
            typename T_IPDElectronSpeciesList>
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc)
        {
            using ForEachElectronSpeciesFillSumFields = pmacc::meta::ForEach<
                T_IPDElectronSpeciesList,
                s_IPD::stage::FillIPDSumFields_Electron<boost::mpl::_1, T_TemperatureFunctor>>;
            using ForEachIonSpeciesFillSumFields = pmacc::meta::
                ForEach<T_IPDIonSpeciesList, s_IPD::stage::FillIPDSumFields_Ion<boost::mpl::_1, T_TemperatureFunctor>>;

            // reset IPD SumFields
            resetSumFields();

            ForEachElectronSpeciesFillSumFields{}(mappingDesc);
            ForEachIonSpeciesFillSumFields{}(mappingDesc);

            s_IPD::stage::CalculateIPDInput<T_numberAtomicPhysicsIonSpecies>()(mappingDesc);
        }

        /** calculate ionization potential depression
         *
         * @param localTemperatureEnergyBox deviceDataBox giving access to the local temperature * k_Boltzman for all
         *  local superCells, in sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2, not weighted
         * @param localZStarBox deviceDataBox giving access to the local z^Star value, = average(q^2) / average(q),
         *  for all local superCells, unitless, not weighted
         * @param localDebyeLengthBox deviceDataBox giving access to the local debye length for all local superCells,
         *  sim.unit.length(), not weighted
         * @param superCellFieldIdx index of superCell in superCellField(without guards)
         *
         * @return unit: eV, not weighted
         */
        template<
            uint8_t T_atomicNumber,
            typename T_LocalDebyeLengthBox,
            typename T_LocalTemperatureEnergyBox,
            typename T_LocalZStarBox>
        HDINLINE static float_X calculateIPD(
            pmacc::DataSpace<simDim> const superCellFieldIdx,
            T_LocalDebyeLengthBox const localDebyeLengthBox,
            T_LocalTemperatureEnergyBox const localTemperatureEnergyBox,
            T_LocalZStarBox const localZStarBox)
        {
            // eV, not weighted
            float_X const temperatureTimesk_Boltzman = localTemperatureEnergyBox(superCellFieldIdx);
            // sim.unit.length(), not weighted
            float_X const debyeLength = localDebyeLengthBox(superCellFieldIdx);

            // unitless, not weighted
            float_X const zStar = localZStarBox(superCellFieldIdx);

            // eV, not weighted
            return T_IPDModel::calculateIPD<T_atomicNumber>(debyeLength, temperatureTimesk_Boltzman, zStar);
        }

        template<typename T_Kernel, uint32_t T_chunkSize, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput)
        {
            auto& localDebyeLengthField
                = *dc.get<s_IPD::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
                    "LocalDebyeLengthField");
            auto& localTemperatureEnergyField
                = *dc.get<s_IPD::localHelperFields::LocalTemperatureEnergyField<picongpu::MappingDesc>>(
                    "LocalTemperatureEnergyField");
            auto& localZStarField
                = *dc.get<s_IPD::localHelperFields::LocalZStarField<picongpu::MappingDesc>>("LocalZStarField");

            PMACC_LOCKSTEP_KERNEL(T_Kernel())
                .template config<T_chunkSize>(mapper.getGridDim())(
                    mapper,
                    kernelInput...,
                    localDebyeLengthField.getDeviceDataBox(),
                    localTemperatureEnergyField.getDeviceDataBox(),
                    localZStarField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
