/* Copyright 2020-2022 Sergei Bastrakov, Brian Marre
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

#include "picongpu/param/physicalConstants.param"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdint>
#include <memory>
#include <utility>


/** @file implements the storage of atomic state and transition data
 *
 * The atomicPhysics step relies on a model of atomic states and transitions for each atomic-
 * Physics ion species. The model's parameters are provided by the user as .txt file of
 * specified format at runtime, external to PIConGPU itself due to license requirements.
 *
 * This file is read at the start of the simulation and stored in an instance of the
 * atomicData Database implemented in this file.
 *
 * too different classes give access to atomic data:
 * - AtomicDataDB ... implements
 *                         * reading of the atomicData input file
 *                         * export to the DataBox for device side use
 *                         * host side storage of atomicData
 * - AtomicDataBox ... deviceSide storage and access to atomicData
 *
 * The atomic data actually consists of 7 different data sets:
 *
 * - list of ionization states (sorted in ascending order of ionization):
 *      [ (ionization energy, [eV]
 *         number of atomicStates,
 *         startIndex of block of atomicStates in atomicState list) ]
 *
 * - list of levels (sorted blockwise by ionization state list)
 *    [(configNumber, [see electronDistribution]
 *      energy respective to ground state of ionization state, [eV]
 *
 *      number of transitions          up b,
 *      startIndex of transition block up b,
 *      number of transitions          down b,
 *      startIndex of transition block down b,
 *
 *      number of transitions          up f,
 *      startIndex of transition block up f,
 *      number of transitions          down f,
 *      startIndex of transition block down f,
 *
 *      number of transitions          up a,
 *      startIndex of transition block up a,
 *      number of transitions          down a,
 *      startIndex of transition block down a)]
 *
 * - bound-bound(b) transitions, list (sorted blockwise by lower State according to state list)
 *    [(collisionalOscillatorStrength,
 *      absorptionOscillatorStrength,
 *      gaunt coefficent 1,
 *      gaunt coefficent 2,
 *      gaunt coefficent 3,
 *      gaunt coefficent 4,
 *      gaunt coefficent 5,
 *      upper state configNumber)]
 *
 * - b reverse loockup list: sorted blockwise by upper State
 *    [ index Transition,
 *      lower state configNumber]
 *
 * - bound-free(f) transitions, list (sorted blockwise by lower State according to state list)
 *    [(phicx coefficent 1,
 *      phicx coefficent 2,
 *      phicx coefficent 3,
 *      phicx coefficent 4,
 *      phicx coefficent 5,
 *      phicx coefficent 6,
 *      phicx coefficent 7,
 *      phicx coefficent 8,
 *      upper state configNumber)]
 *
 * - f reverse loockup list: sorted blockwise by upper State
 *    [ index Transition,
 *      lower state configNumber]
 *
 * - autonomous transitions(a), list (sorted blockwise by lower atomic, according to state list)
 *    [(rate, [1/s]
 *      upper state configNumber)]
 *
 * - f reverse loockup list: sorted blockwise by upper State
 *    [ index Transition,
 *      lower state configNumber]
 *
 * NOTE: - configNumber specifies the number of a state as defined by the configNumber class
 *       - index always refers to a collection index
 *      the configNumber of a given state is always the same, its collection index depends on
 *      input file, => should only be used internally
 */


namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {


        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
