/* Copyright 2022 Brian Marre, Axel Huebl, Sudhir Sharma
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/algorithms/math.hpp>

/** @file implements calculation of rates for bound-bound atomic physics transitions
 *
 * based on the rate calculation of FLYCHK, as extracted by Axel Huebl in the original
 *  flylite prototype.
 *
 * References:
 * - Axel Huebl
 *  first flylite prototype, not published
 *
 *  - R. Mewe.
 *  "Interpolation formulae for the electron impact excitation of ions in
 *  the H-, He-, Li-, and Ne-sequences."
 *  Astronomy and Astrophysics 20, 215 (1972)
 *
 *  - H.-K. Chung, R.W. Lee, M.H. Chen.
 *  "A fast method to generate collisional excitation cross-sections of
 *  highly charged ions in a hot dense matter"
 *  High Energy Dennsity Physics 3, 342-352 (2007)
 */

namespace picongpu::particles::atomicPhysics2::rateCalculation
{
    /** functor providing rates and cross section calculation
     *
     * @tparam T_IonSpecies resolved typename of the ion species
     * @attention atomic data box input data is assumed to be in eV
     */
    template<typename T_IonSpecies>
    class BoundBoundTransitionRates
    {
    }

} // namespace picongpu::particles::atomicPhysics2::rateCalculation
