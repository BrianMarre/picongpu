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

//! @file get TransitionType and TransitionDirection from processClass

#pragma once

#include "picongpu/particles/atomicPhysics2/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionType.hpp"

namespace picongpu::particles::atomicPhysics2::enums
{
    //! processClas to TransitionDataSet
    //@{
    // error case, unknown is always false
    template<ProcessClass T_ProcessClass>
    struct TransitionTypeFor;

    template<>
    struct TransitionTypeFor<ProcessClass::noChange>
    {
        static constexpr TransitionType type = TransitionType::noChange;
        // transition direction not meaningful for noChange transitions, arbitrarily using upward
        static constexpr TransitionDirection direction = TransitionDirection::upward;
    };

    template<>
    struct TransitionTypeFor<ProcessClass::spontaneousDeexcitation>
    {
        static constexpr TransitionType type = TransitionType::boundBound;
        static constexpr TransitionDirection direction = TransitionDirection::downward;
    };

    template<>
    struct TransitionTypeFor<ProcessClass::electronicExcitation>
    {
        static constexpr TransitionType type = TransitionType::boundBound;
        static constexpr TransitionDirection direction = TransitionDirection::upward;
    };

    template<>
    struct TransitionTypeFor<ProcessClass::electronicDeexcitation>
    {
        static constexpr TransitionType type = TransitionType::boundBound;
        static constexpr TransitionDirection direction = TransitionDirection::downward;
    };

    template<>
    struct TransitionTypeFor<ProcessClass::electronicIonization>
    {
        static constexpr TransitionType type = TransitionType::boundFree;
        static constexpr TransitionDirection direction = TransitionDirection::upward;
    };

    template<>
    struct TransitionTypeFor<ProcessClass::autonomousIonization>
    {
        static constexpr TransitionType type = TransitionType::autonomous;
        static constexpr TransitionDirection direction = TransitionDirection::downward;
    };

    template<>
    struct TransitionTypeFor<ProcessClass::fieldIonization>
    {
        static constexpr TransitionType type = TransitionType::boundFree;
        static constexpr TransitionDirection direction = TransitionDirection::upward;
    };

    // pressure ionization does use separate pressure ionization state table, not transitions
    //@}
} // namespace picongpu::particles::atomicPhysics2::enums
