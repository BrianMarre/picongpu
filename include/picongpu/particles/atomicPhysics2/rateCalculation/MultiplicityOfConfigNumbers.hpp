/* Copyright 2022 Brian Marre
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

#pragma once

#include "picongpu/param/atomicPhysics2_Debug.param"
#include "picongpu/particles/atomicPhysics2/rateCalculation/BinomialCoeffcient.hpp"
#include "picongpu/particles/atomicPhysics2/rateCalculation/PowerFunctions.hpp"

//#include "picongpu/simulation_defines.hpp"
#include <pmacc/algorithms/math.hpp>

namespace picongpu::particles::atomicPhysics2::rateCalculation
{
    /** number of physical different atomic configurations for a given configNumber
     *
     * @param configNumber configNumber of atomic state, unitless
     * @return degeneracy, number of physical configurations, unitless
     */
    template<typename T_ConfigNumber>
    HDINLINE float_64 Multiplicity(T_ConfigNumber const configNumber)
    {
        using LevelVector = pmacc::math::Vector<uint8_t, T_ConfigNumber::numberLevels>; // unitless

        // check for overflows
        PMACC_CASSERT(To_high_n_max_overflow_detected_in_Multiplicity, T_ConfigNumber::numberLevels < 12u);

        LevelVector const levelVector = T_ConfigNumber::getLevelVector(configNumber); // unitless

        float_64 result = 1u;
        for(uint8_t i = 0u; i < T_numLevels; i++)
        {
            //  number configurations over number electrons
            result *= binomialCoefficients(
                static_cast<uint8_t>(2u)
                    * Power<uint8_t, 2u>(i + 1), // 2*n^2 ... number of atomic configurations in n-th shell
                levelVector[i]); // unitless
        }

        return result; // unitless
    }

} // namespace picongpu::particles::atomicPhysics2::rateCalculation
