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

//! @file implements conversion from IPD Model to Implementation class

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/Implementation_StewartPyatt.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/Model_StewartPyatt.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/stewartPyatt/temperatureFunctor/TagToFunctor.hpp"


namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    //! default case, needs to be specialized for each model
    template<typename T_Model, typename... T_ModelOption>
    struct ModelToImplementation;

    template<typename T_TemperaturFunctorTag>
    struct ModelToImplementation<stewartPyatt::Model_StewartPyatt<T_TemperaturFunctorTag>, T_TemperaturFunctorTag>
    {
        using type = stewartPyatt::Implementation_StewartPyatt<
            temperatureFunctor::TagToFunctor<T_TemperaturFunctorTag>::type,
            stewartPyatt::Model_StewartPyatt<T_TemperaturFunctorTag>>;
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
