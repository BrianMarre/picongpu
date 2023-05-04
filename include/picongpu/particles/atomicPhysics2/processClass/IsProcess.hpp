/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it andor modify
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

//! @file check if processClass is in processClassesGroup

#pragma once

#include "picongpu/particles/atomicPhysics2/processClass/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics2/processClass/ProcessClassGroup.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::processClass
{
    //! general interface for checking for if a processClass belongs to a processClassGroup
    template<ProcessClassGroup group>
    struct IsProcess
    {
        HDINLINE static constexpr bool check(uint8_t processClass);
    };

    /** processClasses which are based on bound-bound transition data sets,
     *  "picongpu/particles/atomicPhysics2/atomicData/*" */
    template<>
    struct IsProcess<ProcessClassGroup::boundBoundBased>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if((processClass == static_cast<uint8_t>(ProcessClass::spontaneousDeexcitation))
               || (processClass == static_cast<uint8_t>(ProcessClass::electronicExcitation))
               || (processClass == static_cast<uint8_t>(ProcessClass::electronicDeexcitation)))
                return true;
            return false;
        }
    };

    /** processClasses which are based on bound-free transition data sets,
     *  "picongpu/particles/atomicPhysics2/atomicData/*" */
    template<>
    struct IsProcess<ProcessClassGroup::boundFreeBased>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if((processClass == static_cast<uint8_t>(ProcessClass::electronicIonization))
               || (processClass == static_cast<uint8_t>(ProcessClass::fieldIonization)))
                return true;
            return false;
        }
    };

    /** processClasses which are based on autonomous transition data sets,
     *  "picongpu/particles/atomicPhysics2/atomicData/*" */
    template<>
    struct IsProcess<ProcessClassGroup::autonomousBased>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if(processClass == static_cast<uint8_t>(ProcessClass::autonomousIonization))
                return true;
            return false;
        }
    };

    //! processClass which causes ionization
    template<>
    struct IsProcess<ProcessClassGroup::ionizing>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if((processClass == static_cast<uint8_t>(ProcessClass::electronicIonization))
               || (processClass == static_cast<uint8_t>(ProcessClass::autonomousIonization))
               || (processClass == static_cast<uint8_t>(ProcessClass::fieldIonization)))
                return true;
            return false;
        }
    };

    //! processClass describing interaction with free electron
    template<>
    struct IsProcess<ProcessClassGroup::electronicCollisional>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if((processClass
                == static_cast<uint8_t>(
                    picongpu::particles::atomicPhysics2::processClass::ProcessClass ::electronicExcitation))
               || (processClass
                   == static_cast<uint8_t>(
                       picongpu::particles::atomicPhysics2::processClass::ProcessClass ::electronicDeexcitation))
               || (processClass
                   == static_cast<uint8_t>(
                       picongpu::particles::atomicPhysics2::processClass::ProcessClass ::electronicIonization)))
                return true;
            return false;
        }
    };

    //! processClass describing transition with initial state lowerState of transition
    template<>
    struct IsProcess<ProcessClassGroup::upward>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if((processClass
                == static_cast<uint8_t>(
                    picongpu::particles::atomicPhysics2::processClass::ProcessClass ::electronicExcitation)))
                return true;
            return false;
        }
    };

    //! processClass describing interaction with initial state upperState of transition
    template<>
    struct IsProcess<ProcessClassGroup::downward>
    {
        HDINLINE static constexpr bool check(uint8_t processClass)
        {
            if((processClass
                == static_cast<uint8_t>(
                    picongpu::particles::atomicPhysics2::processClass::ProcessClass ::electronicDeexcitation))
               || (processClass
                   == static_cast<uint8_t>(
                       picongpu::particles::atomicPhysics2::processClass::ProcessClass ::spontaneousDeexcitation))
               || (processClass
                   == static_cast<uint8_t>(
                       picongpu::particles::atomicPhysics2::processClass::ProcessClass ::autonomousIonization)))
                return true;
            return false;
        }
    };

} // namespace picongpu::particles::atomicPhysics2::processClass
