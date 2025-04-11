/*
 * Copyright 2025 the original honhimW.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.vectorstore.surrealdb;

import jakarta.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.springframework.ai.vectorstore.filter.Filter.*;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author honhimW
 * @since 2025-04-11
 */

public class SurrealDBFilterExpressionConverterTests {

	static SurrealDBFilterExpressionConverter converter = new SurrealDBFilterExpressionConverter();

	@Test
	void and() {
		String expr = convert(ExpressionType.AND,
				new Expression(ExpressionType.EQ, new Key("flag"), new Value(true)),
				new Expression(ExpressionType.EQ, new Key("flag"), new Value(false)));
		assertThat(expr).isEqualTo("flag = true AND flag = false");
	}

	@Test
	void or() {
		String expr = convert(ExpressionType.OR,
				new Expression(ExpressionType.EQ, new Key("flag"), new Value(true)),
				new Expression(ExpressionType.EQ, new Key("flag"), new Value(false)));
		assertThat(expr).isEqualTo("flag = true OR flag = false");
	}

	@Test
	void eq() {
		String expr = convert(ExpressionType.EQ, new Key("flag"), new Value(false));
		assertThat(expr).isEqualTo("flag = false");
	}

	@Test
	void ne() {
		String expr = convert(ExpressionType.NE, new Key("flag"), new Value(false));
		assertThat(expr).isEqualTo("flag != false");
	}

	@Test
	void gt() {
		String expr = convert(ExpressionType.GT, new Key("flag"), new Value(false));
		assertThat(expr).isEqualTo("flag > false");
	}

	@Test
	void gte() {
		String expr = convert(ExpressionType.GTE, new Key("flag"), new Value(false));
		assertThat(expr).isEqualTo("flag >= false");
	}

	@Test
	void lt() {
		String expr = convert(ExpressionType.LT, new Key("flag"), new Value(false));
		assertThat(expr).isEqualTo("flag < false");
	}

	@Test
	void lte() {
		String expr = convert(ExpressionType.LTE, new Key("flag"), new Value(false));
		assertThat(expr).isEqualTo("flag <= false");
	}

	@Test
	void in() {
		{
			String expr = convert(ExpressionType.IN, new Key("flag"), new Value(true));
			assertThat(expr).isEqualTo("flag IN true");
		}
		{
			String expr = convert(ExpressionType.IN, new Key("flag"), new Value(List.of(true, false)));
			assertThat(expr).isEqualTo("flag IN [true,false]");
		}
	}

	@Test
	void notIn() {
		{
			String expr = convert(ExpressionType.NIN, new Key("flag"), new Value(true));
			assertThat(expr).isEqualTo("flag NOT IN true");
		}
		{
			String expr = convert(ExpressionType.NIN, new Key("flag"), new Value(List.of(true, false)));
			assertThat(expr).isEqualTo("flag NOT IN [true,false]");
		}
	}

	@Test
	void not() {
		{
			Expression expression = new Expression(ExpressionType.NOT, new Expression(ExpressionType.EQ, new Key("flag"), new Value(true)));
			String expr = converter.convertExpression(expression);
			assertThat(expr).isEqualTo("flag != true");
		}
		{
			Expression expression = new Expression(ExpressionType.NOT, new Expression(ExpressionType.IN, new Key("flag"), new Value(List.of(true, false))));
			String expr = converter.convertExpression(expression);
			assertThat(expr).isEqualTo("flag NOT IN [true,false]");
		}
	}

	private String convert(ExpressionType type, Operand left, @Nullable Operand right) {
		Expression expression = new Expression(type, left, right);
		return converter.convertExpression(expression);
	}

}
