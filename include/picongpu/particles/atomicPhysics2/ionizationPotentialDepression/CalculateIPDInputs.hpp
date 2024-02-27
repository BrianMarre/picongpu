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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! @file CalculateIPDInputs sub-stage of atomicPhysics

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/CalculateIPDInputs.kernel"
#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/FillIPDSumFields.kernel"

#include <string>

namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
{
    /** atomic Physics sub-stage for calculating ionization potential depression(IPD) inputs
     *  local temperature, local debye length and z^star of the Stewart-Pyatt IPD model
     *
     * uses two stages
     *(0.) reset sum fields)
     * 1.) reduction over all macro particles of species marked with either isOnlyIPDIon, isAtomicPhysicsIon,
     *     isOnlyIPDElectron or isAtomicPhysicsElectron flag to five sum fields
     * 2.) calculate local temperature, debye length, and z^star as input for each super cell
     */
    template<typename T_TemperatureFunctional>
    struct CalculateIPDInputs
    {
        using SpeciesRepresentingAtomciPhysicsIons =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;
        using SpeciesRepresentingAtomciPhysicsIons =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;
    };
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
