/* Copyright 2019-2022 Brian Marre, Sudhir Sharma
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

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

namespace picongpu::particles::atomicPhysics2::rateCalculation
{
    /** binomial coefficient calculated using partial pascal triangle
     *
     *    should be no problem in flychk data since largest value ~10^10
     *    will become problem if all possible states are used
     *
     * @todo add description of iteration,
     * - algorithm tested against scipy.specia.binomial
     *
     * Source: https://www.tutorialspoint.com/binomial-coefficient-in-cplusplus;
     *  22.11.2019, Sudhir Sharma
     */
    HDINLINE float_64 binomialCoefficients(uint8_t n, uint8_t k)
    {
        // check for limits, no check for < 0 necessary, since uint
        if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CALCULATION_HOT_DEBUG)
            if(n < k)
            {
                printf("invalid call binomial(n,k), n < k");
                return 0.f;
            }

        // reduce necessary steps using symmetry in k
        if(k > (n / 2u))
        {
            k = n - k;
        }

        float_64 result = 1u;

        for(uint8_t i = 1u; i <= k; i++)
        {
            result *= static_cast<float_64>(n - i + 1) / static_cast<float_64>(i);
        }
        return result;
    }

} // namespace picongpu::particles::atomicPhysics2::rateCalculation
